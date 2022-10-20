import pandas as pd
import os
import json
import shutil
import glob
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import librosa


class MOS:
    def __init__(self, source_dir, root_dir):
        self.source_dir = source_dir
        self.root_dir = root_dir

    def last_stage_val_csv(self, dir, stage: str):
        stage_dir = {
            "libri_mask": "mask",
            "mask": "mask",
            "FT": "sup",
            "joint": "sup",
        }[stage]
        csv_name = sorted(
            os.listdir(f"{dir}/train/csv/{stage_dir}"),
            key=lambda x: int(x.split('.')[0].split('=')[-1])
        )[-1]
        return f"{dir}/validate/csv/validate/{csv_name}"

    def average(self, metric):
        outputs = []
        for spk in tqdm(sorted(os.listdir(self.root_dir))):
            output = {}
            for pipeline in sorted(os.listdir(f"{self.root_dir}/{spk}")):
                stages = pipeline.split('-')
                output["speaker"] = spk

                pipeline_dir = f"{self.root_dir}/{spk}/{pipeline}/lightning_logs"
                ver = sorted(
                    os.listdir(pipeline_dir),
                    key=lambda x: int(x.split('_')[-1])
                )[-1]
                dir = f"{pipeline_dir}/{ver}/fit"
                for stage in stages:
                    try:
                        csv_file = self.last_stage_val_csv(dir, stage)
                        df = pd.read_csv(csv_file)
                        score = df[metric].mean()
                        output[(pipeline, stage)] = score
                    except:
                        continue
            outputs.append(output)
        df = pd.DataFrame(outputs).set_index("speaker")#dropna()
        df.columns = pd.MultiIndex.from_tuples(df.columns)
        return df

    def calculate_duration(self, paths):
        total_duration = 0

        def _duration(_paths):
            dur = 0
            for p in _paths:
                dur += librosa.get_duration(filename=p)
            return dur

        for spk in tqdm(paths, desc="Speaker"):
            for key in tqdm(paths[spk], desc="pipeline", leave=False):
                if key == "text":
                    continue
                elif key == "ref":
                    total_duration += _duration(paths[spk][key]) / 4
                elif isinstance(paths[spk][key], list):
                    assert key in {"raw", "recon"}
                    total_duration += _duration(paths[spk][key])
                elif isinstance(paths[spk][key], dict):
                    for stage in paths[spk][key]:
                        assert isinstance(paths[spk][key][stage], list)
                        total_duration += _duration(paths[spk][key][stage])
                else:
                    raise
        return total_duration

    def _copy(self, _paths, dst):
        for fpath in _paths:
            shutil.copy2(fpath, dst)

    def copy_files(self, target, paths, desc=None):
        os.makedirs(target, exist_ok=True)
        for key in tqdm(paths, leave=False, desc=desc):
            if isinstance(paths[key], list):
                dst = f"{target}/{key}"
                os.makedirs(dst, exist_ok=True)
                self._copy(paths[key], dst)
            elif isinstance(paths[key], dict):
                self.copy_files(f"{target}/{key}", paths[key], desc=key)
            else:
                raise (paths, target)

    def _get_fname(self, _paths, val_id=None, k=None, basename=True):
        if val_id is not None:
            outputs = []
            for _path in _paths:
                if _path.split('/')[-1].startswith(val_id):
                    if basename:
                        outputs.append(_path.split('/')[-1])
                    else:
                        outputs.append(_path)
            assert len(outputs) == 1
        elif k is not None:
            _paths = [p.split('/')[-1] for p in _paths]
            outputs = random.sample(_paths, k)
        else:
            raise
        return outputs

    def val_id_to_questions(self, questions, val_id, paths, prefix=None):
        if isinstance(prefix, str):
            paths = paths[prefix]
        elif isinstance(prefix, list):
            for pref in prefix:
                paths = paths[pref]
            prefix = '/'.join(prefix)

        """
        path keys: ["ref", "text", "raw", "recon", ...]
        """

        questions[val_id] = {"test": []}
        for key in tqdm(paths, desc="pipeline", leave=False):
            if key == "text":
                fname = self._get_fname(
                    paths[key], val_id=val_id, basename=False)[0]
                try:
                    lab = open(fname, 'r').read()
                except Exception as e:
                    print(fname)
                    raise
                questions[val_id][key] = lab
            elif key == "ref":
                questions[val_id]["ref"] = [
                    f"{prefix}/ref/{fname}"
                    for fname in self._get_fname(paths[key], k=2)
                ]
            elif isinstance(paths[key], list):
                assert key in {"raw", "recon"}
                fname = self._get_fname(paths[key], val_id=val_id)[0]
                questions[val_id]["test"].append(f"{prefix}/{key}/{fname}")
            elif isinstance(paths[key], dict):
                for stage in paths[key]:
                    assert isinstance(paths[key][stage], list)
                    fname = self._get_fname(paths[key][stage],
                                        val_id=val_id)[0]
                    questions[val_id]["test"].append(f"{prefix}/{key}/{stage}/{fname}")
            else:
                raise
        random.shuffle(questions[val_id]["test"])


class SpeakerMOS(MOS):
    def __init__(self, source_dir, root_dir, val_ids_per_spk=3, val_ids_per_task=4):
        super().__init__(source_dir, root_dir)
        self.val_ids_per_spk = val_ids_per_spk
        self.val_ids_per_task = val_ids_per_task
        mos_csv_fname = f"./mos-{val_ids_per_spk}-{val_ids_per_task}-9.csv"

        if not os.path.exists("./mos_audio") or not os.path.exists(mos_csv_fname):
            self.paths = self.sample(531)
            print(json.dumps(self.paths, indent=4))
        # self.df = self.average(self.paths, 'val_speaker_acc')
        # print(self.df)
        # print(self.df.mean())
        # print(self.df.std())
        if not os.path.exists("./mos_audio"):
            self.copy_files("./mos_audio", self.paths)
        if not os.path.exists(mos_csv_fname):
            outputs = self.generate_mos_csv(self.paths)
            csv_df = pd.DataFrame(outputs)
            csv_df.to_csv(mos_csv_fname, index=False)
            print(outputs)
        else:
            csv_df = pd.read_csv(mos_csv_fname)
        durations = self.csv_durations(csv_df)
        print(min(durations), max(durations))
        sns.histplot(x=durations)
        plt.show()
        #
        # import math
        # q_per_csv = math.floor(50 / 4.2)
        # n_csv = math.ceil(csv_df.shape[0] / q_per_csv)
        # for i in range(n_csv):
        #     row_slice = slice(i * q_per_csv, (i+1) * q_per_csv)
        #     csv_df[row_slice].to_csv(f"{target_dir}/mos_{i}.csv", index=False)
    def csv_durations(self, df, prefix="./mos_audio"):
        durations = []

        def _duration(row):
            dur = 0
            for item in tqdm(row, leave=False):
                if item.endswith(".wav"):
                    dur += librosa.get_duration(filename=f"{prefix}/{item}")
            return dur

        for row in tqdm(df.itertuples()):
            durations.append(_duration(row[1:]))

        return durations

    def get_ref_audio_paths(self, dir):
        outputs = []
        sup_ids = sorted(os.listdir(f"{dir}/train/audio/sup"))
        for sup_id in sup_ids:
            spk = sup_id.split('_')[0]
            ref_wav = glob.glob(f"{self.source_dir}/raw_data/VCTK/*/*/{spk}/{sup_id}.wav")[0]
            outputs.append(ref_wav)
        return outputs

    def get_raw_audio_paths(self, df):
        outputs = []
        for row in df.itertuples():
            val_id = row.val_ids
            spk = val_id.split('_')[0]
            raw_wav = glob.glob(f"{self.source_dir}/raw_data/VCTK/*/*/{spk}/{val_id}.wav")[0]
            outputs.append(raw_wav)
        return outputs

    def get_text_paths(self, df):
        outputs = []
        for row in df.itertuples():
            val_id = row.val_ids
            spk = val_id.split('_')[0]
            text = glob.glob(f"{self.source_dir}/raw_data/VCTK/*/*/{spk}/{val_id}.lab")[0]
            outputs.append(text)
        return outputs

    def get_recon_audio_paths(self, df, dir):
        outputs = []
        audio_dir = f"{dir}/validate/audio/validate"
        for row in df.itertuples():
            val_id = row.val_ids
            recon_wav = f"{audio_dir}/{val_id}/{val_id}.recon.wav"
            outputs.append(recon_wav)
        return outputs

    def get_stage_audio_paths(self, df, dir):
        outputs = []
        audio_dir = f"{dir}/validate/audio/validate"
        for row in df.itertuples():
            epoch = row.epoch
            val_id = row.val_ids
            synth_wav = f"{audio_dir}/{val_id}/{val_id}-epoch={epoch}-batch_idx=0.wav"
            outputs.append(synth_wav)
        return outputs

    def sample(self, random_state=None):
        paths = {}
        for spk in tqdm(sorted(os.listdir(self.root_dir))):
            paths[spk] = {}

            for pipeline in sorted(os.listdir(f"{self.root_dir}/{spk}")):
                paths[spk][pipeline] = {}

                pipeline_dir = f"{self.root_dir}/{spk}/{pipeline}/lightning_logs"
                ver = sorted(
                    os.listdir(pipeline_dir),
                    key=lambda x: int(x.split('_')[-1])
                )[-1]
                dir = f"{pipeline_dir}/{ver}/fit"

                if pipeline == "joint":
                    paths[spk]['ref'] = self.get_ref_audio_paths(dir)

                stages = pipeline.split('-')
                for stage in stages:
                    csv_file = self.last_stage_val_csv(dir, stage)
                    df = pd.read_csv(csv_file)
                    df = df.sample(n=self.val_ids_per_spk, random_state=random_state)

                    if 'raw' not in paths[spk]:
                        paths[spk]['raw'] = self.get_raw_audio_paths(df)
                    if 'text' not in paths[spk]:
                        paths[spk]['text'] = self.get_text_paths(df)
                    if 'recon' not in paths[spk]:
                        paths[spk]['recon'] = self.get_recon_audio_paths(df, dir)
                    paths[spk][pipeline][stage] = self.get_stage_audio_paths(df, dir)
        return paths

    def average(self, paths, metric):
        outputs = []
        for spk in tqdm(paths):
            output = {"speaker": spk}
            for type in paths[spk]:
                if type in {'raw', 'text', 'recon', 'ref'}:
                    continue
                pipeline = type
                pipeline_dir = f"{self.root_dir}/{spk}/{pipeline}/lightning_logs"
                ver = sorted(
                    os.listdir(pipeline_dir),
                    key=lambda x: int(x.split('_')[-1])
                )[-1]
                dir = f"{pipeline_dir}/{ver}/fit"

                for stage in paths[spk][pipeline]:
                    synth_wavs = paths[spk][pipeline][stage]
                    val_ids = [path.split('/')[-1].split('-')[0]
                               for path in synth_wavs]
                    csv_file = self.last_stage_val_csv(dir, stage)
                    df = pd.read_csv(csv_file)
                    df = df[df['val_ids'].isin(val_ids)]
                    # df = df.sample(n=self.val_ids_per_spk, random_state=random_state)
                    score = df[metric].mean()
                    output[(pipeline, stage)] = score
            outputs.append(output)

        df = pd.DataFrame(outputs).set_index("speaker")#dropna()
        df.columns = pd.MultiIndex.from_tuples(df.columns)
        return df

    def generate_val_ids_questions(self, paths):
        questions = {}

        for spk in tqdm(paths, desc="Speaker"):
            val_ids = [p.split('/')[-1].split('.')[0] for p in paths[spk]['raw']]
            for val_id in val_ids:
                self.val_id_to_questions(questions, val_id, paths, prefix=spk)
                # prefix = f"{accent}/{val_id}"
                # paths[accent][val_id]

        return questions

    def val_ids_questions_to_csv(self, questions):
        outputs = []
        val_ids = list(questions.keys())
        random.shuffle(val_ids)

        def _val_id_to_q_dict(q, _id):
            out = {}
            for _i, ref_wav in enumerate(q["ref"]):
                out[f"audio_ref-{_id+1}_{_i+1}"] = ref_wav
            for _i, test_wav in enumerate(q["test"]):
                out[f"audio-{_id+1}_{_i+1}"] = test_wav
            out[f"text-{_id+1}"] = q["text"]
            return out

        for i, val_id in enumerate(val_ids):
            if i % self.val_ids_per_task == 0:
                outputs.append({})
            outputs[-1].update(
                _val_id_to_q_dict(questions[val_id], i%self.val_ids_per_task))
        return outputs

    def generate_mos_csv(self, paths, seed=531):
        random.seed(seed)

        questions = self.generate_val_ids_questions(paths)
        outputs = self.val_ids_questions_to_csv(questions)

        return outputs


if __name__ == "__main__":
    source_dir = "../Meta-TTS"
    root_dir = f"{source_dir}/output/learnable_structured_pipeline"
    # target_dir = "/home/r06942045/myProjects/voice-clone-pruning-mos"
    target_dir = "."
    # metric = 'sparsity'
    metric = 'val_accent_acc'

    # mos = MOS(source_dir, root_dir)
    # df = mos.average(metric)
    # print(df)
    # print(df.mean())
    # print(df.std())

    # spk sample
    spk_mos = SpeakerMOS(source_dir, root_dir,
                         val_ids_per_spk=6,
                         val_ids_per_task=2)
