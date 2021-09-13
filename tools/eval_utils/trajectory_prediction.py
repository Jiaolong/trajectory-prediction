import tqdm
import time
import pickle
import numpy as np
import torch
from lidardet.models import load_data_to_gpu

def eval_trajectory_prediction(cfg, model, dataloader, logger, dist_test=False, save_to_file=False, result_dir=None):
    result_dir.mkdir(parents=True, exist_ok=True)
        
    dataset = dataloader.dataset
    
    if dist_test:
        num_gpus = torch.cuda.device_count()
        local_rank = cfg.local_rank % num_gpus
        model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[local_rank],
                broadcast_buffers=False
        )
    model.eval()
    
    if cfg.local_rank == 0:
        progress_bar = tqdm.tqdm(total=len(dataloader), leave=True, desc='eval', dynamic_ncols=True)
    start_time = time.time()
    result_dicts_list = []
    for i, batch_dict in enumerate(dataloader):
        load_data_to_gpu(batch_dict) 
        with torch.no_grad():
            pred_dicts, ret_dict = model(batch_dict)
       
        result_dicts = dataset.generate_prediction_dicts(batch_dict, pred_dicts, 
                output_path = result_dir if save_to_file else None)
        result_dicts_list.append(result_dicts)

        if cfg.local_rank == 0:
            # progress_bar.set_postfix(disp_dict)
            progress_bar.update()
        
    if cfg.local_rank == 0:
        progress_bar.close()

    sec_per_example = (time.time() - start_time) / len(dataloader.dataset)
    logger.info('Test finished(sec_per_example: %.4f second).' % sec_per_example)

    if cfg.local_rank != 0:
        return {}
    
    dataset.evaluation(result_dicts_list)

    logger.info('Result is save to %s' % result_dir)
    logger.info('****************Evaluation done.*****************')
