# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import time
import os
from itertools import islice
from .logging_utils import Averager
from .common_utils import (
    plot_alignment_to_figure,
    cross_attention_prior_annealing,
    merge_figures
)


def maybe_save_checkpoint(accelerator, lr_scheduler, args):
    if (
        args.current_train_step > args.optim.total_steps
        or args.current_train_step % args.checkpoint.every_steps == 0
    ):
        output_dir = f'checkpoint-{args.mode}-{args.current_train_step}'
        accelerator.save_state(output_dir=output_dir)
        if accelerator.is_main_process:
            torch.save(lr_scheduler.state_dict(), os.path.join(output_dir, "scheduler.bin"))


def maybe_eval_predict(model, dataloader, logger, args, tokenizer, codec_model, accelerator):
    if (
        args.current_train_step > args.optim.total_steps
        or args.current_train_step % args.eval.every_steps == 0
    ):
        model.eval()

        with torch.no_grad():
            eval(model, dataloader, logger, args, tokenizer, accelerator)
            predict(model, dataloader, logger, args, tokenizer, codec_model, accelerator)

        args.last_log = time.time()
        model.train()


def maybe_logging(averager, args, model, optimizer, logger):
    if args.current_train_step % args.logging.every_steps == 0:
        stats = extra_stats(args, model, optimizer)

        averager.update(stats)
        averaged_stats = averager.average()

        logger.log_stats(
            stats=averaged_stats,
            step=args.current_train_step,
            args=args,
            prefix='train/'
        )

        args.last_log = time.time()


def log_stats_to_tensorboard(stats, logger, args, stage="train"):
    for key in stats:
        logger.log_tensorboard(
            tag=f"{stage}/{key}",
            data=stats[key],
            global_step=args.current_train_step,
            type="scalar"
        )

def maybe_grad_clip_and_grad_calc(accelerator, model, args):
    if args.optim.grad_clip > 0:
        grad_l2 = accelerator.clip_grad_norm_(
            parameters=model.parameters(),
            max_norm=args.optim.grad_clip,
            norm_type=2,
        )
    else:
        grad_l2 = None

    if args.logging.grad_l2:
        if grad_l2 is None:
            grad_l2 = (
                sum(p.grad.detach().data.norm(2).item() ** 2 for p in model.parameters()) ** 0.5
            )

        return {'grad_l2': grad_l2}
    else:
        return {}


def extend_acc_stats(acc_stats, stats):
    for key in stats:
        if key not in acc_stats:
            acc_stats[key] = [stats[key]]
        elif isinstance(acc_stats[key], list):
            acc_stats[key].append(stats[key])
        else:
            acc_stats[key] = [acc_stats[key], stats[key]]

    return acc_stats


def average_stats(stats):
    for key in stats:
        if isinstance(stats[key], list):
            stats[key] = sum(stats[key]) / len(stats[key])
        else:
            stats[key] = stats[key].mean().item()

    return stats


def extra_stats(args, model, optimizer):
    stats = {}

    if args.logging.weights_l2:
        weights_l2 = sum(p.detach().norm(2).item() ** 2 for p in model.parameters()) ** 0.5
        stats['weights_l2'] = weights_l2

    stats['lr'] = optimizer.param_groups[0]['lr']
    stats['seconds_per_step'] = (time.time() - args.last_log) / args.logging.every_steps

    return stats


def forward(model, batch, args, calc_acc=False):
    input_ids, decoder_input_ids, labels, cross_attention_prior, \
        input_lengths, decoder_input_lengths, encoder_attention_mask = batch
    
    cross_attention_prior = cross_attention_prior_annealing(
        cross_attention_prior=cross_attention_prior,
        current_train_step=args.current_train_step,
        start_scale_down_step=args.optim.attn_prior_start_scale_down_step,
        end_step=args.optim.attn_prior_end_step
    )
        
    outputs = model(
        input_ids=input_ids,
        attention_mask=encoder_attention_mask,
        decoder_input_ids=decoder_input_ids,
        labels=labels,
        cross_attention_prior=cross_attention_prior,
        input_lengths=input_lengths,
        decoder_input_lengths=decoder_input_lengths
    )

    lm_loss = outputs.loss
    attn_ctc_loss = outputs.attn_ctc_loss

    if args.optim.use_attn_ctc_loss and args.current_train_step >= args.optim.ctc_loss_start_step \
        and args.current_train_step < args.optim.ctc_loss_end_step:
        attn_ctc_loss = args.optim.ctc_loss_scale * attn_ctc_loss
        loss = lm_loss + attn_ctc_loss
    else:
        loss = lm_loss
        
    cross_attention_weights = outputs.decoder_outputs.cross_attn_weights
    averaged_cross_attention_weights = torch.mean(
        cross_attention_weights, dim=[1, 2]).permute(0, 2, 1).detach().cpu().numpy()
    averaged_cross_attention_weights = [
        averaged_cross_attention_weights[index][:input_lengths[index],:decoder_input_lengths[index]]
        for index in range(input_ids.shape[0])
    ]

    stats = {}
    stats['loss'] = loss.detach()
    stats['lm_loss'] = lm_loss.detach()
    stats['attn_ctc_loss'] = attn_ctc_loss.detach()

    if calc_acc:
        acc_mask = ~(labels == -100)
        num_valid_elements = acc_mask.sum()
        correct = (outputs.logits.argmax(-1) == labels) * acc_mask
        num_correct_elements = correct.sum()
        accuracy = num_correct_elements / num_valid_elements
        stats['accuracy'] = accuracy

    return loss, stats, averaged_cross_attention_weights


def eval(model, dataloader, logger, args, tokenizer, accelerator):
    args.last_log = time.time()
    averager = Averager()
    current_alignment_images = 0

    for batch_id, batch in enumerate(dataloader, start=1):
        if batch_id == args.eval.corrected_steps * args.optim.grad_acc:
            break

        _, stats, averaged_alignment = forward(model, batch, args, calc_acc=True)
        gathered_stats = accelerator.gather_for_metrics(stats)

        averaged_stats_across_gpus = average_stats(gathered_stats)
        averager.update(averaged_stats_across_gpus)
        
        if current_alignment_images < args.eval.max_alignment_images:
            for index in range(len(averaged_alignment)):
                figures = []

                averaged_alignment_figure = plot_alignment_to_figure(averaged_alignment[index])
                figures.append(averaged_alignment_figure)

                figure = merge_figures(figures)
                logger.log_tensorboard(
                    tag=f"alignment/{current_alignment_images}", 
                    data=figure, 
                    global_step=args.current_train_step, 
                    type="figure"
                )

                current_alignment_images += 1
        

    averager.update({'time': time.time() - args.last_log})
    averaged_stats = averager.average()

    log_stats_to_tensorboard(averaged_stats, logger, args, stage="eval")

    logger.log_stats(
        stats=averaged_stats,
        step=args.current_train_step,
        args=args,
        prefix='eval/'
    )


def predict(model, dataloader, logger, args, tokenizer, codec_model, accelerator):
    args.last_log = time.time()

    model = accelerator.unwrap_model(model)

    def codec_to_waveform(codec_tokens, codec_lengths):
        codec_tokens = codec_tokens.permute(0, 2, 1)
        waveform, waveform_lengths = codec_model.decode(
            tokens=codec_tokens, tokens_len=codec_lengths)
        return waveform, waveform_lengths
    
    top_p = args.infer.top_p
    top_k = args.infer.top_k
    temperature = args.infer.temperature
    
    predict_samples = 0
    
    for step, batch in enumerate(dataloader):

        input_ids, decoder_input_ids, labels, cross_attention_prior, \
            input_lengths, decoder_input_lengths, encoder_attention_mask = batch
        
        batch_size = input_ids.shape[0]
        
        if args.infer.use_logits_processors:
            top_p_tensor = torch.tensor([top_p] * batch_size, device=batch[0].device)
            top_k_tensor = torch.tensor([top_k] * batch_size, device=batch[0].device)
            temperature_tensor = torch.tensor([temperature] * batch_size, device=input_ids.device)
        else:
            top_p_tensor, top_k_tensor, temperature_tensor = None, None, None
        
        min_decoder_input_length = decoder_input_lengths.min().item()
        max_decoder_input_length = decoder_input_lengths.max().item()
        decoder_prompt_length = min_decoder_input_length // 2
        decoder_prompt_input_ids = decoder_input_ids[:, :decoder_prompt_length, :]
        max_generation_steps = max_decoder_input_length - decoder_prompt_length + 1
        
        generate_outputs = model.generate(
            input_ids=input_ids,
            attention_mask=encoder_attention_mask,
            decoder_prompt_input_ids=decoder_prompt_input_ids,
            max_length=max_generation_steps,
            temperature=temperature_tensor,
            top_k=top_k_tensor,
            top_p=top_p_tensor,
            generation_config=model.generation_config,
        )
        
        decoder_output_ids, generated_valid_lengths, encoder_outputs, decoder_outputs, output_logits, output_last_decoder_hiddens = generate_outputs
        predicted_codec_tokens = decoder_output_ids - 2
        predicted_codec_tokens[predicted_codec_tokens < 0] = 0
        total_valid_predicted_lengths = generated_valid_lengths + decoder_prompt_length - 1
        total_valid_ground_truth_lengths = decoder_input_lengths - 1 # minus 1 to remove the eos token
        
        ground_truth_codec_tokens = labels - 2
        ground_truth_codec_tokens[ground_truth_codec_tokens < 0] = 0
        
        predicted_waveform, predicted_waveform_lengths = codec_to_waveform(
            predicted_codec_tokens, 
            codec_lengths=total_valid_predicted_lengths.reshape(batch_size)
        )
        ground_truth_waveform, ground_truth_waveform_lengths = codec_to_waveform(
            ground_truth_codec_tokens, 
            codec_lengths=total_valid_ground_truth_lengths.reshape(batch_size)
        )
        
        for index in range(batch_size):
            id = step * batch_size + index
            predicted = predicted_waveform[index][:predicted_waveform_lengths[index]]
            ground_truth = ground_truth_waveform[index][:ground_truth_waveform_lengths[index]]
            logger.log_tensorboard(f"predicted/{id}", data=predicted, global_step=args.current_train_step, \
                type="audio", sample_rate=args.codec_model.sample_rate)
            logger.log_tensorboard(f"ground_truth/{id}", data=ground_truth, global_step=args.current_train_step, \
                type="audio", sample_rate=args.codec_model.sample_rate)
        
        predict_samples += batch_size
        if predict_samples >= args.infer.max_predict_samples:
            break
    
    logger.log_stats(
        stats={
            "time": time.time() - args.last_log,
        },
        step=args.current_train_step,
        args=args,
        prefix="test/",
    )


def train(model, train_dataloader, test_dataloader, accelerator, lr_scheduler,
          optimizer, logger, args, tokenizer, codec_model):
    
    model.train()

    train_averager = Averager()

    start_from = (args.current_train_step * args.optim.grad_acc) % len(train_dataloader)

    while args.current_train_step <= args.optim.total_steps:
        # In case there is a remainder from previous epoch, we need to reset the optimizer
        optimizer.zero_grad(set_to_none=True)

        acc_stats = {}

        for batch_id, batch in enumerate(train_dataloader, start=start_from):
            if args.current_train_step > args.optim.total_steps:
                break

            loss, stats, _ = forward(model, batch, args)
            gathered_stats = accelerator.gather_for_metrics(stats)
            # average stats across multiple gpus
            averaged_stats = average_stats(gathered_stats)
            acc_stats = extend_acc_stats(acc_stats, averaged_stats)
            accelerator.backward(loss / args.optim.grad_acc)
            train_averager.update(averaged_stats)

            if batch_id % args.optim.grad_acc == 0:
                stats = maybe_grad_clip_and_grad_calc(accelerator, model, args)
                gathered_stats = accelerator.gather_for_metrics(stats)
                # average stats across multiple gpus
                averaged_stats = average_stats(gathered_stats)
                acc_stats = extend_acc_stats(acc_stats, averaged_stats)
                # average stats across multiple grad acc steps
                averaged_acc_stats = average_stats(acc_stats)
                log_stats_to_tensorboard(averaged_acc_stats, logger, args, stage="train")
                
                acc_stats.clear()
                train_averager.update(averaged_stats)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

                maybe_logging(train_averager, args, model, optimizer, logger)
                maybe_eval_predict(model, test_dataloader, logger, args, tokenizer, codec_model, accelerator)
                maybe_save_checkpoint(accelerator, lr_scheduler, args)

                args.current_train_step += 1
        
        start_from = 1

    maybe_eval_predict(model, test_dataloader, logger, args, tokenizer, codec_model, accelerator)
    maybe_save_checkpoint(accelerator, lr_scheduler, args)