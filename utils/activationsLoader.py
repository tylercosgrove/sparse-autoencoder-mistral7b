import torch
from data.liveDataLoader import LiveDataLoader
from utils.generate import get_input_activations_at_layer
from model import Transformer
import os

"""

Ugly/inefficient dataloader. Would not recommend looking too hard at this.

"""


class ActivationsLoader:
    def __init__(
        self,
        batch_size,
        num_batch,
        mistral_models_path,
        target_layer=24,
        dataloader_batch_size=16,
        d_model=4096,
        act_dir="act-data",
    ):
        self.batch_size = batch_size
        self.num_batch = num_batch
        self.target_layer = target_layer
        self.d_model = d_model
        self.mistral_models_path = mistral_models_path

        self.tokensloader = LiveDataLoader(
            dataloader_batch_size, mistral_models_path=mistral_models_path
        )

        self.act_dir = act_dir

        if self.num_act_files() == 0:
            self.refresh_data()

        self.file_counter = 0
        filenames = os.listdir(act_dir)
        filenames = sorted(filenames)
        filenames = [os.path.join(act_dir, f) for f in filenames]
        self.filenames = filenames

    def new_data(self):
        to_return = self.get_token_acts(self.filenames[self.file_counter])
        self.file_counter += 1
        return to_return

    def num_act_files(self):
        return len(
            [
                f
                for f in os.listdir(self.act_dir)
                if os.path.isfile(os.path.join(self.act_dir, f))
            ]
        )

    def get_token_acts(self, filename):
        tokens = torch.load(filename)
        return tokens

    def refresh_data(self):
        self.delete_pt_files("act-data")
        self.file_counter = 0
        self.filenames = []

        def split_list(lst, n):
            return [lst[i : i + n] for i in range(0, len(lst), n)]

        mistral7b = Transformer.from_folder(self.mistral_models_path).cuda()

        total_acts = torch.empty((0, 4096))
        for i in range(8):
            print(f"batch {i+1}")
            batch = self.tokensloader.next_batch()
            better_batch = [
                sublist for lst in batch for sublist in split_list(lst, 4096)
            ]
            for b in better_batch:
                activations = get_input_activations_at_layer(
                    [b], mistral7b, target_layer=self.target_layer
                ).cpu()
                indices_acts = torch.randperm(activations.shape[0])
                activations = activations[indices_acts]
                total_acts = torch.cat((total_acts, activations))
                del activations

        # shuffle data
        indices = torch.randperm(total_acts.shape[0])
        total_acts = total_acts[indices]
        print(total_acts.shape)

        # save data to files
        split_size = 4096
        output_dir = "act-data"
        splits = torch.split(total_acts, split_size, dim=0)

        for i, split in enumerate(splits):
            file_path = os.path.join(output_dir, f"split_{i:03d}.pt")
            torch.save(split, file_path)

        filenames = os.listdir(output_dir)
        filenames = sorted(filenames)
        filenames = [os.path.join(output_dir, f) for f in filenames]
        self.filenames = filenames

        mistral7b.cpu()
        del mistral7b
        torch.cuda.empty_cache()

    def delete_pt_files(self, dir):
        for filename in os.listdir(dir):
            if filename.endswith(".pt"):
                file_path = os.path.join(dir, filename)
                try:
                    os.remove(file_path)
                except Exception as e:
                    print(f"Error deleting {file_path}: {e}")

    def needs_refresh(self):
        return self.file_counter >= self.num_act_files()
