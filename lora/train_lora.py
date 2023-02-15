import argparse
import math
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.utils.data import Dataset

from accelerate import Accelerator
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    StableDiffusionInpaintPipeline,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
from diffusers.loaders import AttnProcsLayers
from diffusers.models.cross_attention import LoRACrossAttnProcessor
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version
from PIL import Image, ImageDraw
from torchvision.prototype import transforms  # TODO: migrate to pytorch 2.0 for easy transform impl
# from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
import time

import hfai

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.10.0.dev0")

ip = os.environ['MASTER_IP']
port = os.environ['MASTER_PORT']
world_size = int(os.environ['WORLD_SIZE'])  # 机器个数
rank = int(os.environ['RANK'])  # 当前机器编号
local_rank = int(os.environ['LOCAL_RANK'])
gpus = torch.cuda.device_count()  # 每台机器的GPU个数

def prepare_mask_and_masked_image(image, mask):
    """
    args:
        image: PIL image
        mask: torch tensor

    TODO: reuse instance images to avoid redundancy
    """
    image = np.array(image.convert("RGB"))
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image).to(dtype=torch.float32) / 127.5 - 1.0

    # mask = np.array(mask.convert("L"))
    # mask = mask.astype(np.float32) / 255.0
    mask = mask[None]
    mask[mask < 0.5] = 0
    mask[mask >= 0.5] = 1
    # mask = torch.from_numpy(mask)

    masked_image = image * (mask < 0.5)

    return mask, masked_image

def difference_grid(images, masks, inpaints, num):
    w, h = images[0].size
    grid = Image.new("RGB", size=(w * 3, h * num))
    for i in range(num):
        grid.paste(images[i], box=(0 * w, i * h))
        grid.paste(masks[i], box=(1 * w, i * h))
        grid.paste(inpaints[i], box=(2 * w, i * h))
    return grid

def parse_args():
    parser = argparse.ArgumentParser(description="Human inpainting with lora.")
    parser.add_argument(
        "--name",
        type=str,
        default="test_lora",
        required=True,
        help="A name to specific the work",
    )
    parser.add_argument(
        "--pretrained_model_name",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--pretrained_model_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--instance_data_dir",
        type=str,
        default=None,
        required=True,
        help="A folder containing the training data of instance images.",
    )
    parser.add_argument(
        "--instance_prompt",
        type=str,
        default="a photo of a coco person",
        help="The prompt with identifier specifying the instance",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="text-inversion-model",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--center_crop", action="store_true", help="Whether to center crop images before resizing to resolution"
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=4, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="no",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose"
            "between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10."
            "and an Nvidia Ampere GPU."
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    
    parser.add_argument(
        "--eval_step",
        type=int,
        default=50,
        help="Validation every n epoches"
    )
    parser.add_argument(
        "--eval_data_root",
        type=str,
        default=None,
        help="eval data root"
    )
    parser.add_argument(
        "--eval_num",
        type=int,
        default=4
    )

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    if args.instance_data_dir is None:
        raise ValueError("You must specify a train data directory.")

    return args


class HumanDataset(Dataset):
    """
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images and the tokenizes prompts.
    """

    def __init__(
        self,
        instance_data_root,
        instance_prompt,
        # tokenizer,
        class_data_root=None,
        class_prompt=None,
        size=512,
        center_crop=False,
    ):
        self.size = size
        self.center_crop = center_crop
        # self.tokenizer = tokenizer

        self.instance_data_root = Path(instance_data_root)
        if not self.instance_data_root.exists():
            raise ValueError("Instance images root doesn't exists.")

        self.instance_images_path = list(Path(instance_data_root).iterdir())
        self.num_instance_images = len(self.instance_images_path)
        self.instance_prompt = instance_prompt
        self._length = self.num_instance_images


        self.class_data_root = None

        self.image_transforms_resize_and_crop = transforms.Compose(
            [
                transforms.RandomShortestSize(size+3), # TODO: avoid min size failure
                transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size),
            ]
        )

        self.image_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

        self.mask_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        example = {}
        # get mask image
        mask_path = self.instance_images_path[index % self.num_instance_images]
        mask = Image.open(mask_path)

        # TODO: now the instance path is hard coded to be under masks parent dir, should change
        image_path = str(mask_path).replace("masks", "images")
        instance_image = Image.open(image_path)
        if not instance_image.mode == "RGB":
            instance_image = instance_image.convert("RGB")
        
        instance_image, mask = self.image_transforms_resize_and_crop(instance_image, mask)

        example["PIL_images"] = instance_image
        example["masks"] = self.mask_transforms(mask)
        example["instance_images"] = self.image_transforms(instance_image)
        
        # pre load the fix prompt to avoid loading clip
        # example["instance_prompt_ids"] = self.tokenizer(
        #     self.instance_prompt,
        #     padding="do_not_pad",
        #     truncation=True,
        #     max_length=self.tokenizer.model_max_length,
        # ).input_ids

        return example

class MyPipeline(StableDiffusionInpaintPipeline):        
    def prepare_mask_latents(self, mask, masked_image, batch_size, height, width, dtype, device, generator, do_classifier_free_guidance):
        return super().prepare_mask_latents(mask, masked_image, batch_size, height, width, dtype, device, generator, do_classifier_free_guidance)

def main():
    print(f"{rank}: start")
    args = parse_args()
    logging_dir = Path(f"{args.output_dir}", args.logging_dir)

    dist.init_process_group(backend='nccl',
                            init_method=f'tcp://{ip}:{port}',
                            world_size=world_size,
                            rank=rank)
    torch.cuda.set_device(local_rank)
    print("{}: ".format(dist.get_rank()) + "finish distributed")

    # Handle the repository creation
    if dist.get_rank() == 0:
        if args.output_dir is not None:
            os.makedirs(f"{args.output_dir}", exist_ok=True)

    # Load the tokenizer
    if args.tokenizer_name:
        tokenizer = CLIPTokenizer.from_pretrained(args.tokenizer_name)
    elif args.pretrained_model_path:
        tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_path, subfolder="tokenizer")
        
    weight_dtype = torch.float32
    if args.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif args.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Load models and create wrapper for stable diffusion
    print("{}: ".format(dist.get_rank()) + "start to load models")
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_path, subfolder="text_encoder").to("cuda", dtype=weight_dtype)
    print("{}: ".format(dist.get_rank()) + "finish text_encoder")
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name, subfolder="vae").to("cuda", dtype=weight_dtype)
    print("{}: ".format(dist.get_rank()) + "finish vae")
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_path, subfolder="unet").to("cuda", dtype=weight_dtype)
    print("{}: ".format(dist.get_rank()) + "finish unet")

    '''
        Lora part
    '''    
    text_encoder.requires_grad_(False)
    vae.requires_grad_(False)
    unet.requires_grad_(False)
    
    lora_attn_procs = {}
    for name in unet.attn_processors.keys():
        cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
        if name.startswith("mid_block"):
            hidden_size = unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = unet.config.block_out_channels[block_id]

        lora_attn_procs[name] = LoRACrossAttnProcessor(
            hidden_size=hidden_size, cross_attention_dim=cross_attention_dim
        )

    unet.set_attn_processor(lora_attn_procs)
    lora_layers = AttnProcsLayers(unet.attn_processors).to("cuda", dtype=weight_dtype)
    lora_layers = DistributedDataParallel(lora_layers, device_ids=[local_rank])
    ''''''

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * world_size
        )

    # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )

        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    # disable text_encoder
    optimizer = optimizer_class(
        lora_layers.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_path, subfolder="scheduler")
    
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )
    print(f"{rank}: finish initialize models, optimizer, schedulers")
    
    train_dataset = HumanDataset(
        instance_data_root=args.instance_data_dir,
        instance_prompt=args.instance_prompt,
        class_data_root=None,
        class_prompt=None,
        # tokenizer=tokenizer,
        size=args.resolution,
        center_crop=args.center_crop,
    )
    train_sampler = DistributedSampler(train_dataset)

    def collate_fn(examples):
        # prompt input disabled
        # input_ids = [example["instance_prompt_ids"] for example in examples]
        pixel_values = [example["instance_images"] for example in examples]

        masks = []
        masked_images = []

        # TODO: move this to the dataloader for potential speedup
        # or at least batchify this (loop is slow)
        for example in examples:
            pil_image = example["PIL_images"]
            mask = example["masks"]
            # prepare mask and masked image
            mask, masked_image = prepare_mask_and_masked_image(pil_image, mask)

            masks.append(mask)
            masked_images.append(masked_image)

        pixel_values = torch.stack(pixel_values)
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

        # input_ids = tokenizer.pad({"input_ids": input_ids}, padding=True, return_tensors="pt").input_ids
        masks = torch.stack(masks)
        masked_images = torch.stack(masked_images)
        # prompt input disabled
        # batch = {"input_ids": input_ids, "pixel_values": pixel_values, "masks": masks, "masked_images": masked_images}
        batch = {"pixel_values": pixel_values, "masks": masks, "masked_images": masked_images}
        return batch

    train_dataloader = torch.utils.data.DataLoader(
        dataset=train_dataset, sampler=train_sampler, batch_size=args.train_batch_size, shuffle=False, collate_fn=collate_fn, drop_last=True
    )
    print(f"{rank}: finish initialize data")

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if dist.get_rank() == 0:
        tracker = SummaryWriter(log_dir=logging_dir, comment="human_inpaint_lora")

    # pre load the text encoder result and release text_encoder
    input_ids = tokenizer(
         args.instance_prompt,
         padding="do_not_pad",
         truncation=True,
         max_length=tokenizer.model_max_length,
    ).input_ids
    # make this batch size one. TODO: repeat batch size times if cannot broadcast
    input_ids = [input_ids] * args.train_batch_size
    input_ids = tokenizer.pad({"input_ids": input_ids}, padding=True, return_tensors="pt").input_ids
    encoder_hidden_states = text_encoder(input_ids.to("cuda"))[0].cuda().to(dtype=weight_dtype)

    # Train!
    total_batch_size = args.train_batch_size * world_size * args.gradient_accumulation_steps

    if dist.get_rank() == 0:
        print("***** Running training *****")
        print(f"  Num examples = {len(train_dataset)}")
        print(f"  Num batches each epoch = {len(train_dataloader)}")
        print(f"  Num Epochs = {args.num_train_epochs}")
        print(f"  Instantaneous batch size per device = {args.train_batch_size}")
        print(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        print(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        print(f"  Total optimization steps = {args.max_train_steps}")
        # Only show the progress bar once on each machine.
        progress_bar = tqdm(range(args.max_train_steps), disable=dist.get_rank() != 0)
        progress_bar.set_description("Steps")
        global_step = 0

    print(f"{rank}: start training")
    for epoch in range(args.num_train_epochs):
        for step, batch in enumerate(train_dataloader):
            # Convert images to latent space

            latents = vae.encode(batch["pixel_values"].to(dtype=weight_dtype).to("cuda")).latent_dist.sample()
            latents = latents * 0.18215

            # Convert masked images to latent space
            masked_latents = vae.encode(
                batch["masked_images"].reshape(batch["pixel_values"].shape).to(dtype=weight_dtype).to("cuda")
            ).latent_dist.sample()
            masked_latents = masked_latents * 0.18215

            masks = batch["masks"].to("cuda")
            # resize the mask to latents shape as we concatenate the mask to the latents
            mask = torch.stack(
                [
                    torch.nn.functional.interpolate(mask, size=(args.resolution // 8, args.resolution // 8))
                    for mask in masks
                ]
            )
            mask = mask.reshape(-1, 1, args.resolution // 8, args.resolution // 8).to(dtype=weight_dtype)

            # Sample noise that we'll add to the latents
            noise = torch.randn_like(latents, device="cuda")
            bsz = latents.shape[0]
            # Sample a random timestep for each image
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
            timesteps = timesteps.long()

            # Add noise to the latents according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            # concatenate the noised latents with the mask and the masked latents
            latent_model_input = torch.cat([noisy_latents, mask, masked_latents], dim=1)

            # Predict the noise residual
            noise_pred = unet(latent_model_input, timesteps, encoder_hidden_states).sample

            # Get the target for loss depending on the prediction type
            if noise_scheduler.config.prediction_type == "epsilon":
                target = noise
            elif noise_scheduler.config.prediction_type == "v_prediction":
                target = noise_scheduler.get_velocity(latents, noise, timesteps)
            else:
                raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

            loss = F.mse_loss(noise_pred.float(), target.float(), reduction="mean")

            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if dist.get_rank() == 0:
                progress_bar.update(1)
                global_step += 1

            if dist.get_rank() == 0:
                logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
                progress_bar.set_postfix(**logs)
                print(f"step {global_step}: ", logs)
                tracker.add_scalars(f"human_inpaint/{args.name}",logs, global_step=global_step)
                tracker.add_scalars("human_inpaint_loss", {f"{args.name}_loss":loss.detach().item()}, global_step=global_step)
                tracker.add_scalars("human_inpaint_lr", {f"{args.name}_lr":lr_scheduler.get_last_lr()[0]}, global_step=global_step)
                
            if step >= args.max_train_steps:
                break
                
            if (dist.get_rank() == 0 and (global_step % args.eval_step == 0) and args.eval_data_root != None):
                print(f"Starting validation, step {global_step}")
                
                pipe = MyPipeline.from_pretrained(
                    args.pretrained_model_path,
                    vae = vae,
                    unet = unet,
                    text_encoder = text_encoder
                )
                pipe = pipe.to("cuda")
                prompt = args.instance_prompt
                instance_images_path = list(Path(args.eval_data_root).iterdir())
                
                image_transforms_resize_and_crop = transforms.Compose(
                    [
                        transforms.RandomShortestSize(512+3), 
                        transforms.CenterCrop(512)
                    ]
                )
                image_transforms = transforms.Compose(
                    [
                        transforms.ToTensor(),
                        transforms.Normalize([0.5], [0.5]),
                    ]
                )
                mask_transforms = transforms.Compose(
                    [
                        transforms.ToTensor(),
                    ]
                )
                images, masks, inpaints = [], [], []
                
                for index, image_path in enumerate(tqdm(instance_images_path)):
                    if (index >= args.eval_num): break
                    
                    image = Image.open(image_path)
                    # image = transforms.Resize((512,512))(image)
                    mask_image = Image.open(str(image_path).replace('images','masks'))
                    if not image.mode == "RGB":
                        image = image.convert("RGB")
                    if not mask_image.mode == "L":
                        mask_image = mask_image.convert("L")

                    image, mask_image = image_transforms_resize_and_crop(image, mask_image)
                    img = image_transforms(image).to("cuda")
                    mask = mask_transforms(mask_image).to("cuda")
                    inpaint = pipe(prompt=prompt, image=img, mask_image=mask).images[0]
                    
                    images.append(image)
                    masks.append(mask_image)
                    inpaints.append(inpaint)

                
                grid = difference_grid(images, masks, inpaints, len(images))
                grid = transforms.ToTensor()(grid)
                tracker.add_image("human_inpaint/{}".format(args.name), grid, global_step=global_step)
                
                for index, param in enumerate(lora_layers.module.parameters()):
                    tracker.add_histogram("human_inpaint/{}/layer_{}".format(args.name, index), param.data, global_step=global_step)

                del pipe
                torch.cuda.empty_cache()
                
                print(f"Finish validation, step {global_step}")
                
            if ((step + 1) % args.eval_step == 0 and args.eval_data_root != None):
                dist.barrier()
            
            if dist.get_rank() == 0 and hfai.receive_suspend_command():
                if not os.path.exists(os.path.join(f"{args.output_dir}", './latest/')): os.mkdir(os.path.join(f"{args.output_dir}", './latest/'))
                torch.save(lora_layers.module.state_dict(), os.path.join(os.path.join(f"{args.output_dir}", './latest/'), './lora_layers.ckpt'))
                time.sleep(5)
                hfai.go_suspend()
            
        # print("wait for rank 0")
        dist.barrier()

    print(f"{local_rank}: finish train")
    
    # Create the pipeline using using the trained modules and save it.
    if dist.get_rank() == 0:
        if not os.path.exists(os.path.join(f"{args.output_dir}", './final/')): os.mkdir(os.path.join(f"{args.output_dir}", './final/'))
        torch.save(lora_layers.module.state_dict(), os.path.join(os.path.join(f"{args.output_dir}", './final/'), './lora_layers.ckpt'))

if __name__ == "__main__":
    #test_dataloader()
    main()
