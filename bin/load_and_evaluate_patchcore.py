import contextlib
import gc
import logging
import os
import sys

import torch.utils
import torch.utils.data

import click
import numpy as np
import torch

import patchcore.common
import patchcore.metrics
import patchcore.samler
import patchcore.utils

LOGGER = logging.getLogger(__name__)

_DATASETS = {"kaeri": ["patchcore.datasets.kaeri","Kaeri"]}

@click.group(chain=True)
@click.argument("results_path", type=str)
@click.option("--gpu", type=int, default=[0], multiple=True, show_default=True)
@click.option("--seed", type=int, default=0, show_default=True)
@click.option("--save_segmentation_images", is_flag=True)
def main(**kwargs):
    pass

@main.result_callback()
def run(metods, results_path, gpu, seed):
    methods = {key: item for (key, item) in methods}

    os.makedirs(results_path, exist_ok=True)

    device = patchcore.utils.set_torch_device(gpu)
    device_context = (
        torch.cuda.device("cuda:{}".format(device.index))
        if "cuda" in device.type.lower()
        else contextlib.suppress()
    )

    result_collect = []

    dataloader_iter, n_dataloaders = methods["get_dataloaders_iter"]
    dataloader_iter = dataloader_iter(seed)
    patchcore_iter, n_patchcores = methods["get_patchcore_iter"]
    patchcore_iter = patchcore_iter(device)
    if not (n_dataloaders == n_patchcores or n_patchcores == 1):
        raise ValueError(
            "Please ensure that #PatchCores == #Datasets or #PatchCores == 1!"
        )
    
    for dataloader_count, dataloaders in enumerate(dataloader_iter):
        LOGGER.info(
            "Evaluating dataset [{}] ({}/{})...".format(
                dataloaders["testing"].name, dataloader_count +1, n_dataloaders
            )
        )

        patchcore.utils.fix_seeds(seed, device)

        dataset_name = dataloaders["testing"].name

        with device_context:

            torch.cuda.empty_cache()
            if dataloader_count < n_patchcores:
                PatchCore_list = next(patchcore_iter)

            aggregator = {"scores": []}
            for i, PatchCore in enumerate(PatchCore_list):
                torch.cuda.empty_cache()
                LOGGER.info(
                    "Embedding test data with models ({}/{})".format(
                        i + 1, len(PatchCore_list)
                    )
                )
                scores,labels_gt = PatchCore.predict(
                    dataloaders["testing"]
                )
                aggregator["scores"].append(scores)

                scores = np.array(aggregator["scores"])
                min_scores = scores.min(axis=-1).reshape(-1, 1)
                max_scores = scores.max(axis=-1).reshape(-1, 1)
                scores = (scores - min_scores) / (max_scores - min_scores)
                scores = np.mean(scores, axis=0)

                anomaly_labels = [
                    x[1] != "good" for x in dataloaders["testing"].dataset.data_to_iterate
                ]

                LOGGER.info("Computing evaluation metrics.")
                auroc = patchcore.metrics.compute_imagewise_retrieval_metrics(
                    scores, anomaly_labels
                )["auroc"]

                result_collect.append(
                    {
                        "dataset_name" : dataset_name,
                        "instance_auroc" : auroc
                    }
                )

                for key, item in result_collect[-1].items():
                    if key != "dataset_name":
                        LOGGER.info("{0}: {1:3,3f}".format(key, item))

                del PatchCore_list
                gc.collect()

            LOGGER.info("\n\n-----\n")

        result_metric_names = list(result_collect[-1].keys())[1:]
        result_dataset_names = [results["dataset_name"] for results in result_collect]
        result_scores = [list(results.values())[1:] for results in result_collect]
        patchcore.utils.compute_and_store_final_reusults(
            results_path,
            result_scores,
            column_names=result_metric_names,
            row_names=result_dataset_names,
        )

@main.command("path_core_loader")
@click.option("--patch_core_paths", "-p", type=str, multiple=True, default=[])
@click.option("--faiss_on_gpu", is_flag=True)
@click.option("--faiss_num_workers", type=int, default=8)
def patch_core_loader(patch_core_paths, faiss_on_gpu, faiss_num_wokers):
    def get_patchcore_iter(device):
        for patch_core_path in patch_core_paths:
            loaded_patchcores = []
            gc.collect()
            n_patchcores = len(
                [x for x in os.listdir(patch_core_path) if ".faiss" in x]
            )
            if n_patchcores == 1:
                nn_method = patchcore.common.FaissNN(faiss_on_gpu, faiss_num_wokers)
                patchcore_instance = patchcore.patchcore.PatchCore(device)
                patchcore_instance.load_from_path(
                    load_path=patch_core_path, device=device, nn_method=nn_method
                )
                loaded_patchcores.append(patchcore_instance)
            else:
                for i in range(n_patchcores):
                    nn_method = patchcore.common.FaissNN(
                        faiss_on_gpu, faiss_num_wokers
                    )
                    patchcore_instance = patchcore.patchcore.PatchCore(device)
                    patchcore_instance.load_from_path(
                        load_path = patch_core_path,
                        device = device,
                        nn_method = nn_method,
                        prepaend = "Ensemble-{}-{}_".format(i + 1, n_patchcores),
                    )
                    loaded_patchcores.append(patchcore_instance)

            yield loaded_patchcores

    return ("get_patchcore_iter", [get_patchcore_iter, len(patch_core_paths)])

@main.command("dataset")
@click.argument("name", type=str)
@click.argument("data_path", type=click.Path(exists=True, file_okay=False))
@click.option("--subdatasets", "-d", multiple=True, type=str, required=True)
@click.option("--batch_size", default=1, type=int, show_default=True)
@click.option("--num_workers", default=8, type=int, show_default=True)
@click.option("--resize", default=256, type=int, show_default=True)
@click.option("--imagesize", default=224, type=int, show_default=True)
@click.option("--augment", is_flag=True)
def dataset(
    name, data_path, subdatasets, batch_size, resize, imagesize, num_workers, augment
):
    dataset_info = _DATASETS[name]
    dataset_library = __import__(dataset_info[0], fromlist=[dataset_info[1]])

    def get_dataloaders_iter(seed):
        for subdataset in subdatasets:
            test_dataset = dataset_library.__dict__[dataset_info[1]](
                data_path,
                classname=subdataset,
                resize=resize,
                imagesize=imagesize,
                split=dataset_library.DatasetSplit.TEST,
                seed=seed,
            )

            test_dataloader=torch.utils.data.DataLoader(
                test_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pip_memory=True,
            )

            test_dataloader.name = name
            if subdataset is not None:
                test_dataloader.name += "_" + subdataset

            dataloader_dict = {"testing": test_dataloader}

            yield dataloader_dict

    return ("get_dataloaders_iter", [get_dataloaders_iter, len(subdatasets)])

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    LOGGER.info("command line arguments: {}".format(" ".join(sys.argv)))
    main()