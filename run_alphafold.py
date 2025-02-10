# Copyright 2024 DeepMind Technologies Limited
#
# AlphaFold 3 source code is licensed under CC BY-NC-SA 4.0. To view a copy of
# this license, visit https://creativecommons.org/licenses/by-nc-sa/4.0/
#
# To request access to the AlphaFold 3 model parameters, follow the process set
# out at https://github.com/google-deepmind/alphafold3. You may only use these
# if received directly from Google. Use is subject to terms of use available at
# https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md

"""AlphaFold 3 structure prediction script.

AlphaFold 3 source code is licensed under CC BY-NC-SA 4.0. To view a copy of
this license, visit https://creativecommons.org/licenses/by-nc-sa/4.0/

To request access to the AlphaFold 3 model parameters, follow the process set
out at https://github.com/google-deepmind/alphafold3. You may only use these
if received directly from Google. Use is subject to terms of use available at
https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md
"""

from collections.abc import Callable, Sequence
import csv
import dataclasses
import datetime
import functools
import multiprocessing
import os
import pathlib
import shutil
import string
import textwrap
import time
import typing
from typing import overload, List, Tuple

from absl import app
from absl import flags
from alphafold3.common import folding_input
from alphafold3.common import resources
from alphafold3.constants import chemical_components
import alphafold3.cpp
from alphafold3.data import featurisation
from alphafold3.data import pipeline
from alphafold3.jax.attention import attention
from alphafold3.model import features
from alphafold3.model import model
from alphafold3.model import params
from alphafold3.model import post_processing
from alphafold3.model.components import utils
import haiku as hk
import jax
from jax import numpy as jnp
import numpy as np
import psutil
import GPUtil
from concurrent.futures import ProcessPoolExecutor
import torch


_HOME_DIR = pathlib.Path(os.environ.get('HOME'))
_DEFAULT_MODEL_DIR = _HOME_DIR / 'models'
_DEFAULT_DB_DIR = _HOME_DIR / 'public_databases'


# Input and output paths.
_JSON_PATH = flags.DEFINE_string(
    'json_path',
    None,
    'Path to the input JSON file.',
)
_INPUT_DIR = flags.DEFINE_string(
    'input_dir',
    None,
    'Path to the directory containing input JSON files.',
)
_OUTPUT_DIR = flags.DEFINE_string(
    'output_dir',
    None,
    'Path to a directory where the results will be saved.',
)
MODEL_DIR = flags.DEFINE_string(
    'model_dir',
    _DEFAULT_MODEL_DIR.as_posix(),
    'Path to the model to use for inference.',
)

# Control which stages to run.
_RUN_DATA_PIPELINE = flags.DEFINE_bool(
    'run_data_pipeline',
    True,
    'Whether to run the data pipeline on the fold inputs.',
)
_RUN_INFERENCE = flags.DEFINE_bool(
    'run_inference',
    True,
    'Whether to run inference on the fold inputs.',
)

# Binary paths.
_JACKHMMER_BINARY_PATH = flags.DEFINE_string(
    'jackhmmer_binary_path',
    shutil.which('jackhmmer'),
    'Path to the Jackhmmer binary.',
)
_NHMMER_BINARY_PATH = flags.DEFINE_string(
    'nhmmer_binary_path',
    shutil.which('nhmmer'),
    'Path to the Nhmmer binary.',
)
_HMMALIGN_BINARY_PATH = flags.DEFINE_string(
    'hmmalign_binary_path',
    shutil.which('hmmalign'),
    'Path to the Hmmalign binary.',
)
_HMMSEARCH_BINARY_PATH = flags.DEFINE_string(
    'hmmsearch_binary_path',
    shutil.which('hmmsearch'),
    'Path to the Hmmsearch binary.',
)
_HMMBUILD_BINARY_PATH = flags.DEFINE_string(
    'hmmbuild_binary_path',
    shutil.which('hmmbuild'),
    'Path to the Hmmbuild binary.',
)

# Database paths.
DB_DIR = flags.DEFINE_multi_string(
    'db_dir',
    (_DEFAULT_DB_DIR.as_posix(),),
    'Path to the directory containing the databases. Can be specified multiple'
    ' times to search multiple directories in order.',
)

_SMALL_BFD_DATABASE_PATH = flags.DEFINE_string(
    'small_bfd_database_path',
    '${DB_DIR}/bfd-first_non_consensus_sequences.fasta',
    'Small BFD database path, used for protein MSA search.',
)
_MGNIFY_DATABASE_PATH = flags.DEFINE_string(
    'mgnify_database_path',
    '${DB_DIR}/mgy_clusters_2022_05.fa',
    'Mgnify database path, used for protein MSA search.',
)
_UNIPROT_CLUSTER_ANNOT_DATABASE_PATH = flags.DEFINE_string(
    'uniprot_cluster_annot_database_path',
    '${DB_DIR}/uniprot_all_2021_04.fa',
    'UniProt database path, used for protein paired MSA search.',
)
_UNIREF90_DATABASE_PATH = flags.DEFINE_string(
    'uniref90_database_path',
    '${DB_DIR}/uniref90_2022_05.fa',
    'UniRef90 database path, used for MSA search. The MSA obtained by '
    'searching it is used to construct the profile for template search.',
)
_NTRNA_DATABASE_PATH = flags.DEFINE_string(
    'ntrna_database_path',
    '${DB_DIR}/nt_rna_2023_02_23_clust_seq_id_90_cov_80_rep_seq.fasta',
    'NT-RNA database path, used for RNA MSA search.',
)
_RFAM_DATABASE_PATH = flags.DEFINE_string(
    'rfam_database_path',
    '${DB_DIR}/rfam_14_9_clust_seq_id_90_cov_80_rep_seq.fasta',
    'Rfam database path, used for RNA MSA search.',
)
_RNA_CENTRAL_DATABASE_PATH = flags.DEFINE_string(
    'rna_central_database_path',
    '${DB_DIR}/rnacentral_active_seq_id_90_cov_80_linclust.fasta',
    'RNAcentral database path, used for RNA MSA search.',
)
_PDB_DATABASE_PATH = flags.DEFINE_string(
    'pdb_database_path',
    '${DB_DIR}/mmcif_files',
    'PDB database directory with mmCIF files path, used for template search.',
)
_SEQRES_DATABASE_PATH = flags.DEFINE_string(
    'seqres_database_path',
    '${DB_DIR}/pdb_seqres_2022_09_28.fasta',
    'PDB sequence database path, used for template search.',
)

# Number of CPUs to use for MSA tools.
_JACKHMMER_N_CPU = flags.DEFINE_integer(
    'jackhmmer_n_cpu',
    min(multiprocessing.cpu_count(), 8),
    'Number of CPUs to use for Jackhmmer. Default to min(cpu_count, 8). Going'
    ' beyond 8 CPUs provides very little additional speedup.',
)
_NHMMER_N_CPU = flags.DEFINE_integer(
    'nhmmer_n_cpu',
    min(multiprocessing.cpu_count(), 8),
    'Number of CPUs to use for Nhmmer. Default to min(cpu_count, 8). Going'
    ' beyond 8 CPUs provides very little additional speedup.',
)

# Template search configuration.
_MAX_TEMPLATE_DATE = flags.DEFINE_string(
    'max_template_date',
    '2021-09-30',  # By default, use the date from the AlphaFold 3 paper.
    'Maximum template release date to consider. Format: YYYY-MM-DD. All '
    'templates released after this date will be ignored.',
)

_CONFORMER_MAX_ITERATIONS = flags.DEFINE_integer(
    'conformer_max_iterations',
    None,  # Default to RDKit default parameters value.
    'Optional override for maximum number of iterations to run for RDKit '
    'conformer search.',
)

# JAX inference performance tuning.
_JAX_COMPILATION_CACHE_DIR = flags.DEFINE_string(
    'jax_compilation_cache_dir',
    None,
    'Path to a directory for the JAX compilation cache.',
)
_GPU_DEVICE = flags.DEFINE_integer(
    'gpu_device',
    0,
    'Optional override for the GPU device to use for inference. Defaults to the'
    ' 1st GPU on the system. Useful on multi-GPU systems to pin each run to a'
    ' specific GPU.',
)
_BUCKETS = flags.DEFINE_list(
    'buckets',
    # pyformat: disable
    ['256', '512', '768', '1024', '1280', '1536', '2048', '2560', '3072',
     '3584', '4096', '4608', '5120'],
    # pyformat: enable
    'Strictly increasing order of token sizes for which to cache compilations.'
    ' For any input with more tokens than the largest bucket size, a new bucket'
    ' is created for exactly that number of tokens.',
)
_FLASH_ATTENTION_IMPLEMENTATION = flags.DEFINE_enum(
    'flash_attention_implementation',
    default='triton',
    enum_values=['triton', 'cudnn', 'xla'],
    help=(
        "Flash attention implementation to use. 'triton' and 'cudnn' uses a"
        ' Triton and cuDNN flash attention implementation, respectively. The'
        ' Triton kernel is fastest and has been tested more thoroughly. The'
        " Triton and cuDNN kernels require Ampere GPUs or later. 'xla' uses an"
        ' XLA attention implementation (no flash attention) and is portable'
        ' across GPU devices.'
    ),
)
_NUM_RECYCLES = flags.DEFINE_integer(
    'num_recycles',
    10,
    'Number of recycles to use during inference.',
    lower_bound=1,
)
_NUM_DIFFUSION_SAMPLES = flags.DEFINE_integer(
    'num_diffusion_samples',
    5,
    'Number of diffusion samples to generate.',
    lower_bound=1,
)
_NUM_SEEDS = flags.DEFINE_integer(
    'num_seeds',
    None,
    'Number of seeds to use for inference. If set, only a single seed must be'
    ' provided in the input JSON. AlphaFold 3 will then generate random seeds'
    ' in sequence, starting from the single seed specified in the input JSON.'
    ' The full input JSON produced by AlphaFold 3 will include the generated'
    ' random seeds. If not set, AlphaFold 3 will use the seeds as provided in'
    ' the input JSON.',
    lower_bound=1,
)

# Output controls.
_SAVE_EMBEDDINGS = flags.DEFINE_bool(
    'save_embeddings',
    False,
    'Whether to save the final trunk single and pair embeddings in the output.',
)

# 添加新的命令行参数
_MAIN_GPU = flags.DEFINE_integer(
    'main_gpu',
    0,
    'GPU device to use for the main process.',
)
_WORKER_GPU = flags.DEFINE_integer(
    'worker_gpu',
    1,
    'GPU device to use for worker processes.',
)


def make_model_config(
    *,
    flash_attention_implementation: attention.Implementation = 'triton',
    num_diffusion_samples: int = 5,
    num_recycles: int = 10,
    return_embeddings: bool = False,
) -> model.Model.Config:
  """Returns a model config with some defaults overridden."""
  config = model.Model.Config()
  config.global_config.flash_attention_implementation = (
      flash_attention_implementation
  )
  config.heads.diffusion.eval.num_samples = num_diffusion_samples
  config.num_recycles = num_recycles
  config.return_embeddings = return_embeddings
  return config


class ModelRunner:
  """Helper class to run structure prediction stages."""

  def __init__(
      self,
      config: model.Model.Config,
      device: jax.Device,
      model_dir: pathlib.Path,
  ):
    self._model_config = config
    self._device = device
    self._model_dir = model_dir

  @functools.cached_property
  def model_params(self) -> hk.Params:
    """Loads model parameters from the model directory."""
    return params.get_model_haiku_params(model_dir=self._model_dir)

  @functools.cached_property
  def _model(
      self,
  ) -> Callable[[jnp.ndarray, features.BatchDict], model.ModelResult]:
    """Loads model parameters and returns a jitted model forward pass."""

    @hk.transform
    def forward_fn(batch):
      return model.Model(self._model_config)(batch)

    return functools.partial(
        jax.jit(forward_fn.apply, device=self._device), self.model_params
    )

  def run_inference(
      self, featurised_example: features.BatchDict, rng_key: jnp.ndarray
  ) -> model.ModelResult:
    """Computes a forward pass of the model on a featurised example."""
    featurised_example = jax.device_put(
        jax.tree_util.tree_map(
            jnp.asarray, utils.remove_invalidly_typed_feats(featurised_example)
        ),
        self._device,
    )

    result = self._model(rng_key, featurised_example)
    result = jax.tree.map(np.asarray, result)
    result = jax.tree.map(
        lambda x: x.astype(jnp.float32) if x.dtype == jnp.bfloat16 else x,
        result,
    )
    result = dict(result)
    identifier = self.model_params['__meta__']['__identifier__'].tobytes()
    result['__identifier__'] = identifier
    return result

  def extract_structures(
      self,
      batch: features.BatchDict,
      result: model.ModelResult,
      target_name: str,
  ) -> list[model.InferenceResult]:
    """Generates structures from model outputs."""
    return list(
        model.Model.get_inference_result(
            batch=batch, result=result, target_name=target_name
        )
    )

  def extract_embeddings(
      self,
      result: model.ModelResult,
  ) -> dict[str, np.ndarray] | None:
    """Extracts embeddings from model outputs."""
    embeddings = {}
    if 'single_embeddings' in result:
      embeddings['single_embeddings'] = result['single_embeddings']
    if 'pair_embeddings' in result:
      embeddings['pair_embeddings'] = result['pair_embeddings']
    return embeddings or None


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class ResultsForSeed:
  """Stores the inference results (diffusion samples) for a single seed.

  Attributes:
    seed: The seed used to generate the samples.
    inference_results: The inference results, one per sample.
    full_fold_input: The fold input that must also include the results of
      running the data pipeline - MSA and templates.
    embeddings: The final trunk single and pair embeddings, if requested.
  """

  seed: int
  inference_results: Sequence[model.InferenceResult]
  full_fold_input: folding_input.Input
  embeddings: dict[str, np.ndarray] | None = None


def predict_structure(
    fold_input: folding_input.Input,
    model_runner: ModelRunner,
    main_gpu: int,
    worker_gpu: int,
    buckets: Sequence[int] | None = None,
    conformer_max_iterations: int | None = None,
) -> Sequence[ResultsForSeed]:
    """在主进程中处理特征化，在GPU 1上只运行推理"""
    results = []
    
    # 在主进程（GPU 0）上进行特征化
    print(f'Featurising data with {len(fold_input.rng_seeds)} seed(s)...')
    featurisation_start_time = time.time()
    ccd = chemical_components.cached_ccd(user_ccd=fold_input.user_ccd)
    featurised_examples = featurisation.featurise_input(
        fold_input=fold_input,
        buckets=buckets,
        ccd=ccd,
        verbose=True,
        conformer_max_iterations=conformer_max_iterations,
    )
    print(
        f'Featurising data took {time.time() - featurisation_start_time:.2f} seconds.'
    )
    
    # 创建进程池用于推理，但不预先设置 GPU
    ctx = multiprocessing.get_context('spawn')
    with ctx.Pool(1, maxtasksperchild=1) as pool:
        for seed, example in zip(fold_input.rng_seeds, featurised_examples):
            print(f'Running inference for seed {seed}...')
            
            # 为每个seed创建rng_key
            rng_key = jax.random.PRNGKey(seed)
            
            # 在子进程中再决定使用哪个 GPU
            result = pool.apply(
                run_inference_process,
                args=(
                    example,
                    rng_key,
                    model_runner._model_config,
                    model_runner._model_dir,
                    main_gpu,
                    worker_gpu,
                    True,
                )
            )
            
            # 在主进程（GPU 0）上提取结构和embeddings
            print(f'Extracting output structure samples with seed {seed}...')
            inference_results = model_runner.extract_structures(
                batch=example, result=result, target_name=fold_input.name
            )
            embeddings = model_runner.extract_embeddings(result)
            
            results.append(
                ResultsForSeed(
                    seed=seed,
                    inference_results=inference_results,
                    full_fold_input=fold_input,
                    embeddings=embeddings,
                )
            )
            
            # 清理内存
            del result, inference_results, embeddings
            jax.clear_caches()
    
    return results

def write_fold_input_json(
    fold_input: folding_input.Input,
    output_dir: os.PathLike[str] | str,
) -> None:
  """Writes the input JSON to the output directory."""
  os.makedirs(output_dir, exist_ok=True)
  path = os.path.join(output_dir, f'{fold_input.sanitised_name()}_data.json')
  print(f'Writing model input JSON to {path}')
  with open(path, 'wt') as f:
    f.write(fold_input.to_json())


def write_outputs(
    all_inference_results: Sequence[ResultsForSeed],
    output_dir: os.PathLike[str] | str,
    job_name: str,
) -> None:
  """Writes outputs to the specified output directory."""
  ranking_scores = []
  max_ranking_score = None
  max_ranking_result = None

  output_terms = (
      pathlib.Path(alphafold3.cpp.__file__).parent / 'OUTPUT_TERMS_OF_USE.md'
  ).read_text()

  os.makedirs(output_dir, exist_ok=True)
  for results_for_seed in all_inference_results:
    seed = results_for_seed.seed
    for sample_idx, result in enumerate(results_for_seed.inference_results):
      sample_dir = os.path.join(output_dir, f'seed-{seed}_sample-{sample_idx}')
      os.makedirs(sample_dir, exist_ok=True)
      post_processing.write_output(
          inference_result=result, output_dir=sample_dir
      )
      ranking_score = float(result.metadata['ranking_score'])
      ranking_scores.append((seed, sample_idx, ranking_score))
      if max_ranking_score is None or ranking_score > max_ranking_score:
        max_ranking_score = ranking_score
        max_ranking_result = result

    if embeddings := results_for_seed.embeddings:
      embeddings_dir = os.path.join(output_dir, f'seed-{seed}_embeddings')
      os.makedirs(embeddings_dir, exist_ok=True)
      post_processing.write_embeddings(
          embeddings=embeddings, output_dir=embeddings_dir
      )

  if max_ranking_result is not None:  # True iff ranking_scores non-empty.
    post_processing.write_output(
        inference_result=max_ranking_result,
        output_dir=output_dir,
        # The output terms of use are the same for all seeds/samples.
        terms_of_use=output_terms,
        name=job_name,
    )
    # Save csv of ranking scores with seeds and sample indices, to allow easier
    # comparison of ranking scores across different runs.
    with open(os.path.join(output_dir, 'ranking_scores.csv'), 'wt') as f:
      writer = csv.writer(f)
      writer.writerow(['seed', 'sample', 'ranking_score'])
      writer.writerows(ranking_scores)


@overload
def process_fold_input(
    fold_input: folding_input.Input,
    data_pipeline_config: pipeline.DataPipelineConfig | None,
    model_runner: None,
    output_dir: os.PathLike[str] | str,
    buckets: Sequence[int] | None = None,
) -> folding_input.Input:
  ...


@overload
def process_fold_input(
    fold_input: folding_input.Input,
    data_pipeline_config: pipeline.DataPipelineConfig | None,
    model_runner: ModelRunner,
    output_dir: os.PathLike[str] | str,
    buckets: Sequence[int] | None = None,
) -> Sequence[ResultsForSeed]:
  ...


def replace_db_dir(path_with_db_dir: str, db_dirs: Sequence[str]) -> str:
  """Replaces the DB_DIR placeholder in a path with the given DB_DIR."""
  template = string.Template(path_with_db_dir)
  if 'DB_DIR' in template.get_identifiers():
    for db_dir in db_dirs:
      path = template.substitute(DB_DIR=db_dir)
      if os.path.exists(path):
        return path
    raise FileNotFoundError(
        f'{path_with_db_dir} with ${{DB_DIR}} not found in any of {db_dirs}.'
    )
  if not os.path.exists(path_with_db_dir):
    raise FileNotFoundError(f'{path_with_db_dir} does not exist.')
  return path_with_db_dir


def process_fold_input(
    fold_input: folding_input.Input,
    data_pipeline_config: pipeline.DataPipelineConfig | None,
    model_runner: ModelRunner | None,
    output_dir: os.PathLike[str] | str,
    buckets: Sequence[int] | None = None,
    conformer_max_iterations: int | None = None,
    main_gpu: int = 0,
    worker_gpu: int = 1,
) -> folding_input.Input | Sequence[ResultsForSeed]:
  """Runs data pipeline and/or inference on a single fold input.

  Args:
    fold_input: Fold input to process.
    data_pipeline_config: Data pipeline config to use. If None, skip the data
      pipeline.
    model_runner: Model runner to use. If None, skip inference.
    output_dir: Output directory to write to.
    buckets: Bucket sizes to pad the data to, to avoid excessive re-compilation
      of the model. If None, calculate the appropriate bucket size from the
      number of tokens. If not None, must be a sequence of at least one integer,
      in strictly increasing order. Will raise an error if the number of tokens
      is more than the largest bucket size.
    conformer_max_iterations: Optional override for maximum number of iterations
      to run for RDKit conformer search.
    main_gpu: GPU device to use for the main process.
    worker_gpu: GPU device to use for worker processes.

  Returns:
    The processed fold input, or the inference results for each seed.

  Raises:
    ValueError: If the fold input has no chains.
  """
  print(f'\nRunning fold job {fold_input.name}...')

  if not fold_input.chains:
    raise ValueError('Fold input has no chains.')

  if os.path.exists(output_dir) and os.listdir(output_dir):
    new_output_dir = (
        f'{output_dir}_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}'
    )
    print(
        f'Output will be written in {new_output_dir} since {output_dir} is'
        ' non-empty.'
    )
    output_dir = new_output_dir
  else:
    print(f'Output will be written in {output_dir}')

  if data_pipeline_config is None:
    print('Skipping data pipeline...')
  else:
    print('Running data pipeline...')
    fold_input = pipeline.DataPipeline(data_pipeline_config).process(fold_input)

  write_fold_input_json(fold_input, output_dir)
  if model_runner is None:
    print('Skipping model inference...')
    output = fold_input
  else:
    print(
        f'Predicting 3D structure for {fold_input.name} with'
        f' {len(fold_input.rng_seeds)} seed(s)...'
    )
    all_inference_results = predict_structure(
        fold_input=fold_input,
        model_runner=model_runner,
        buckets=buckets,
        conformer_max_iterations=conformer_max_iterations,
        main_gpu=main_gpu,
        worker_gpu=worker_gpu,
    )
    print(f'Writing outputs with {len(fold_input.rng_seeds)} seed(s)...')
    write_outputs(
        all_inference_results=all_inference_results,
        output_dir=output_dir,
        job_name=fold_input.sanitised_name(),
    )
    output = all_inference_results

  print(f'Fold job {fold_input.name} done, output written to {output_dir}\n')
  return output


def get_available_gpu_memory(gpu_id: int) -> float:
    """获取指定GPU的可用显存(GB)"""
    try:
        gpu = GPUtil.getGPUs()[gpu_id]
        return gpu.memoryFree / 1024  # 转换为GB
    except Exception:
        return 0

def estimate_batch_size(available_memory: float) -> int:
    """根据可用显存估算每个批次可处理的输入数量
    
    Args:
        available_memory: 可用显存大小(GB)
    
    Returns:
        每个批次可处理的输入数量
    """
    # 假设每个输入平均需要2GB显存,可以根据实际情况调整
    memory_per_input = 2
    return max(1, int(available_memory / memory_per_input))

def process_batch(
    fold_inputs: List[folding_input.Input],
    data_pipeline_config: pipeline.DataPipelineConfig,
    model_dir: str,
    output_dir: str,
    main_gpu: int,
    worker_gpu: int,
    buckets: Tuple[int, ...],
    flash_attention_implementation: str,
    num_diffusion_samples: int,
    num_recycles: int,
    save_embeddings: bool,
    conformer_max_iterations: int | None = None,
) -> None:
    """处理一批输入文件"""
    print(f"Worker process starting with GPU {worker_gpu}")
    os.environ['CUDA_VISIBLE_DEVICES'] = str(worker_gpu)
    
    devices = jax.local_devices(backend='gpu')
    print(f"Worker process found devices: {devices}")
    if not devices:
        raise RuntimeError(f"No GPU devices found for worker process")
    
    model_runner = ModelRunner(
        config=make_model_config(
            flash_attention_implementation=typing.cast(
                attention.Implementation, flash_attention_implementation
            ),
            num_diffusion_samples=num_diffusion_samples,
            num_recycles=num_recycles,
            return_embeddings=save_embeddings,
        ),
        device=devices[0],
        model_dir=pathlib.Path(model_dir),
    )
    
    for fold_input in fold_inputs:
        print(f"Worker process processing input: {fold_input.name}")
        process_fold_input(
            fold_input=fold_input,
            data_pipeline_config=data_pipeline_config,
            model_runner=model_runner,
            output_dir=os.path.join(output_dir, fold_input.sanitised_name()),
            buckets=buckets,
            conformer_max_iterations=conformer_max_iterations,
            main_gpu=main_gpu,
            worker_gpu=worker_gpu,
        )
    print(f"Worker process completed batch")

def get_gpu_mapping():
    """获取实际GPU ID到CUDA_VISIBLE_DEVICES映射的GPU ID的转换"""
    cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES')
    if cuda_visible_devices is not None:
        # 如果设置了CUDA_VISIBLE_DEVICES,需要进行GPU ID映射
        visible_gpus = [int(x) for x in cuda_visible_devices.split(',')]
        print(f"CUDA_VISIBLE_DEVICES is set to: {cuda_visible_devices}")
        print(f"Visible GPUs: {visible_gpus}")
        return {i: visible_gpus[i] for i in range(len(visible_gpus))}
    return None

def get_gpu_utilization(gpu_id: int) -> float:
    """获取指定GPU的使用率"""
    try:
        gpu = GPUtil.getGPUs()[gpu_id]
        return gpu.memoryUtil * 100  # 转换为百分比
    except Exception:
        return 0

def select_gpu_for_operation() -> jax.Device:
    """根据GPU使用情况选择合适的GPU设备"""
    gpu0_util = get_gpu_utilization(_MAIN_GPU.value)
    gpu1_util = get_gpu_utilization(_WORKER_GPU.value)
    
    print(f"Current GPU utilization - GPU {_MAIN_GPU.value}: {gpu0_util:.1f}%, GPU {_WORKER_GPU.value}: {gpu1_util:.1f}%")
    
    devices = jax.local_devices(backend='gpu')
    # 如果主GPU使用率超过80%，且备用GPU使用率较低，则使用备用GPU
    if gpu0_util > 80 and gpu1_util < gpu0_util:
        print(f"Switching to GPU {_WORKER_GPU.value} for next operation")
        return devices[1]
    return devices[0]

def get_gpu_memory_info(gpu_id: int) -> Tuple[float, float]:
    """获取指定GPU的显存使用情况
    
    Returns:
        Tuple[float, float]: (已用显存GB, 总显存GB)
    """
    try:
        gpu = GPUtil.getGPUs()[gpu_id]
        # GPUtil 返回的单位是MB，直接除以1024转换为GB
        return (gpu.memoryUsed / 1024.0, gpu.memoryTotal / 1024.0)
    except Exception:
        return (0, 0)

def select_gpu_for_inference(main_gpu: int, worker_gpu: int) -> int:
    """选择显存占用率较低的GPU用于推理"""
    gpu0_used, gpu0_total = get_gpu_memory_info(main_gpu)
    gpu1_used, gpu1_total = get_gpu_memory_info(worker_gpu)
    
    # 计算可用显存
    gpu0_free = gpu0_total - gpu0_used
    gpu1_free = gpu1_total - gpu1_used
    
    print(f"GPU {main_gpu} memory: {gpu0_used:.1f}GB/{gpu0_total:.1f}GB (Free: {gpu0_free:.1f}GB)")
    print(f"GPU {worker_gpu} memory: {gpu1_used:.1f}GB/{gpu1_total:.1f}GB (Free: {gpu1_free:.1f}GB)")
    
    # 需要至少8GB可用显存
    MIN_REQUIRED_MEMORY = 8.0  # 确保是浮点数
    
    if gpu0_free >= MIN_REQUIRED_MEMORY and gpu0_free > gpu1_free:
        print(f"Selected GPU {main_gpu} with {gpu0_free:.1f}GB free memory")
        return main_gpu
    elif gpu1_free >= MIN_REQUIRED_MEMORY:
        print(f"Selected GPU {worker_gpu} with {gpu1_free:.1f}GB free memory")
        return worker_gpu
    else:
        raise RuntimeError(
            f"No GPU has enough free memory (need at least {MIN_REQUIRED_MEMORY}GB). "
            f"GPU {main_gpu}: {gpu0_free:.1f}GB free, GPU {worker_gpu}: {gpu1_free:.1f}GB free"
        )

def create_gpu_process(gpu_id: int) -> None:
    """在指定GPU上创建预留进程并预留少量显存
    
    Args:
        gpu_id: GPU ID
    """
    # 设置环境变量
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    
    try:
        # 预留128MB显存
        placeholder = torch.empty(128 * 1024 * 1024, dtype=torch.int8, device='cuda')
        placeholder.record_stream(torch.cuda.current_stream())
        print(f"Created placeholder process on GPU {gpu_id}")
        
        # 保持进程运行
        while True:
            time.sleep(1)
    except Exception as e:
        print(f"Error creating placeholder process on GPU {gpu_id}: {e}")

def run_inference_process(
    featurised_example: features.BatchDict,
    rng_key: jnp.ndarray,
    model_config: model.Model.Config,
    model_dir: pathlib.Path,
    main_gpu: int,
    worker_gpu: int,
    is_worker_gpu: bool = False,
) -> model.ModelResult:
    """在显存占用较低的GPU上运行推理"""
    try:
        # 1. 在JAX初始化前检查显存状态
        gpu0_used, gpu0_total = get_gpu_memory_info(main_gpu)
        gpu1_used, gpu1_total = get_gpu_memory_info(worker_gpu)
        gpu0_free = gpu0_total - gpu0_used
        gpu1_free = gpu1_total - gpu1_used
        
        print(f"Initial GPU memory status:")
        print(f"GPU {main_gpu}: {gpu0_used:.1f}GB/{gpu0_total:.1f}GB (Free: {gpu0_free:.1f}GB)")
        print(f"GPU {worker_gpu}: {gpu1_used:.1f}GB/{gpu1_total:.1f}GB (Free: {gpu1_free:.1f}GB)")
        
        # 2. 检查是否有足够显存
        MIN_REQUIRED_MEMORY = 8.0
        if max(gpu0_free, gpu1_free) < MIN_REQUIRED_MEMORY:
            raise RuntimeError(
                f"No GPU has enough free memory before JAX initialization (need at least {MIN_REQUIRED_MEMORY}GB). "
                f"GPU {main_gpu}: {gpu0_free:.1f}GB free, GPU {worker_gpu}: {gpu1_free:.1f}GB free"
            )
        
        # 3. 选择显存较多的GPU
        target_gpu = main_gpu if gpu0_free > gpu1_free else worker_gpu
        print(f"Selected GPU {target_gpu} with {max(gpu0_free, gpu1_free):.1f}GB free memory")
        
        # 4. 清理JAX缓存
        jax.clear_caches()
        
        # 5. 设置环境变量 - 在JAX初始化前设置
        os.environ['CUDA_VISIBLE_DEVICES'] = str(target_gpu)
        os.environ.update({
            'XLA_PYTHON_CLIENT_MEM_FRACTION': '0.95',
            'XLA_PYTHON_CLIENT_PREALLOCATE': 'true',
            'XLA_PYTHON_CLIENT_ALLOCATOR': 'platform',
            'XLA_FORCE_HOST_PLATFORM_DEVICE_COUNT': '1',
            'TF_FORCE_GPU_ALLOW_GROWTH': 'false',
            'XLA_PYTHON_CLIENT_MEM_LIMIT_MB': '14000',
            'XLA_PYTHON_CLIENT_DEVICE_PRIORITY': '0',
        })
        
        # 6. 重新初始化JAX
        import importlib
        importlib.reload(jax.lib)
        importlib.reload(jax)
        
        # 7. 验证JAX设备并再次检查显存
        devices = jax.devices('gpu')
        if not devices:
            raise RuntimeError("No GPU devices found")
        print(f"JAX initialized with devices: {devices}")
        
        used, total = get_gpu_memory_info(target_gpu)
        free = total - used
        print(f"GPU {target_gpu} memory after JAX init: {used:.1f}GB/{total:.1f}GB (Free: {free:.1f}GB)")
        
        if free < MIN_REQUIRED_MEMORY:
            raise RuntimeError(f"Insufficient GPU memory after JAX initialization: only {free:.1f}GB available")
        
        # 8. 创建模型实例
        with jax.default_device(devices[0]):
            inference_model = ModelRunner(
                config=model_config,
                device=devices[0],
                model_dir=model_dir
            )
            print("Successfully created model runner")
            
            # 检查模型加载后的显存
            used, total = get_gpu_memory_info(target_gpu)
            free = total - used
            print(f"GPU {target_gpu} memory after model creation: {used:.1f}GB/{total:.1f}GB (Free: {free:.1f}GB)")
            
            # 9. 运行推理
            print("Starting inference...")
            result = inference_model.run_inference(featurised_example, rng_key)
            print("Inference completed successfully")
            
            # 检查推理后的显存
            used, total = get_gpu_memory_info(target_gpu)
            free = total - used
            print(f"GPU {target_gpu} memory after inference: {used:.1f}GB/{total:.1f}GB (Free: {free:.1f}GB)")
        
        # 10. 确保结果在CPU上
        result = jax.tree_util.tree_map(
            lambda x: np.array(x) if isinstance(x, (np.ndarray, jnp.ndarray)) else x,
            result
        )
        
        return result
        
    except Exception as e:
        print(f"Error in inference process: {str(e)}")
        print(f"Current CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")
        print(f"JAX devices: {jax.devices()}")
        print(f"Current JAX backend: {jax.default_backend()}")
        used, total = get_gpu_memory_info(target_gpu)
        print(f"GPU {target_gpu} memory at error: {used:.1f}GB/{total:.1f}GB")
        raise

def main(_):
    # 在两个GPU上创建预留进程
    gpu0_process = multiprocessing.Process(
        target=create_gpu_process,
        args=(_MAIN_GPU.value,)
    )
    gpu1_process = multiprocessing.Process(
        target=create_gpu_process,
        args=(_WORKER_GPU.value,)
    )
    
    gpu0_process.start()
    gpu1_process.start()
    
    # 等待进程创建完成
    time.sleep(2)
    
    try:
        # 主进程使用 GPU 0，允许预分配显存以提高性能
        os.environ['CUDA_VISIBLE_DEVICES'] = str(_MAIN_GPU.value)
        # GPU 0 的显存管理：允许预分配，但限制使用量
        os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.8'
        os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'true'
        
        if _JAX_COMPILATION_CACHE_DIR.value is not None:
            jax.config.update(
                'jax_compilation_cache_dir', _JAX_COMPILATION_CACHE_DIR.value
            )

        if _JSON_PATH.value is None == _INPUT_DIR.value is None:
            raise ValueError(
                'Exactly one of --json_path or --input_dir must be specified.'
            )

        if not _RUN_INFERENCE.value and not _RUN_DATA_PIPELINE.value:
            raise ValueError(
                'At least one of --run_inference or --run_data_pipeline must be'
                ' set to true.'
            )

        if _INPUT_DIR.value is not None:
            fold_inputs = folding_input.load_fold_inputs_from_dir(
                pathlib.Path(_INPUT_DIR.value)
            )
            print(f"Found {len(list(fold_inputs))} input files in {_INPUT_DIR.value}")
        elif _JSON_PATH.value is not None:
            fold_inputs = folding_input.load_fold_inputs_from_path(
                pathlib.Path(_JSON_PATH.value)
            )
            print(f"Loading input from {_JSON_PATH.value}")
        else:
            raise AssertionError(
                'Exactly one of --json_path or --input_dir must be specified.'
            )

        # 验证输入文件是否为空
        fold_input_list = list(fold_inputs)
        if not fold_input_list:
            raise ValueError(f"No valid input files found in the specified location")
        
        print(f"Total number of inputs to process: {len(fold_input_list)}")

        # Make sure we can create the output directory before running anything.
        try:
            os.makedirs(_OUTPUT_DIR.value, exist_ok=True)
        except OSError as e:
            print(f'Failed to create output directory {_OUTPUT_DIR.value}: {e}')
            raise

        if _RUN_INFERENCE.value:
            # Fail early on incompatible devices, but only if we're running inference.
            gpu_devices = jax.local_devices(backend='gpu')
            if gpu_devices:
                compute_capability = float(
                    gpu_devices[_GPU_DEVICE.value].compute_capability
                )
                if compute_capability < 6.0:
                    raise ValueError(
                        'AlphaFold 3 requires at least GPU compute capability 6.0 (see'
                        ' https://developer.nvidia.com/cuda-gpus).'
                    )
                elif 7.0 <= compute_capability < 8.0:
                    xla_flags = os.environ.get('XLA_FLAGS')
                    required_flag = '--xla_disable_hlo_passes=custom-kernel-fusion-rewriter'
                    if not xla_flags or required_flag not in xla_flags:
                        raise ValueError(
                            'For devices with GPU compute capability 7.x (see'
                            ' https://developer.nvidia.com/cuda-gpus) the ENV XLA_FLAGS must'
                            f' include "{required_flag}".'
                        )
                    if _FLASH_ATTENTION_IMPLEMENTATION.value != 'xla':
                        raise ValueError(
                            'For devices with GPU compute capability 7.x (see'
                            ' https://developer.nvidia.com/cuda-gpus) the'
                            ' --flash_attention_implementation must be set to "xla".'
                        )

        notice = textwrap.wrap(
            'Running AlphaFold 3. Please note that standard AlphaFold 3 model'
            ' parameters are only available under terms of use provided at'
            ' https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md.'
            ' If you do not agree to these terms and are using AlphaFold 3 derived'
            ' model parameters, cancel execution of AlphaFold 3 inference with'
            ' CTRL-C, and do not use the model parameters.',
            break_long_words=False,
            break_on_hyphens=False,
            width=80,
        )
        print('\n' + '\n'.join(notice) + '\n')

        if _RUN_DATA_PIPELINE.value:
            expand_path = lambda x: replace_db_dir(x, DB_DIR.value)
            max_template_date = datetime.date.fromisoformat(_MAX_TEMPLATE_DATE.value)
            data_pipeline_config = pipeline.DataPipelineConfig(
                jackhmmer_binary_path=_JACKHMMER_BINARY_PATH.value,
                nhmmer_binary_path=_NHMMER_BINARY_PATH.value,
                hmmalign_binary_path=_HMMALIGN_BINARY_PATH.value,
                hmmsearch_binary_path=_HMMSEARCH_BINARY_PATH.value,
                hmmbuild_binary_path=_HMMBUILD_BINARY_PATH.value,
                small_bfd_database_path=expand_path(_SMALL_BFD_DATABASE_PATH.value),
                mgnify_database_path=expand_path(_MGNIFY_DATABASE_PATH.value),
                uniprot_cluster_annot_database_path=expand_path(
                    _UNIPROT_CLUSTER_ANNOT_DATABASE_PATH.value
                ),
                uniref90_database_path=expand_path(_UNIREF90_DATABASE_PATH.value),
                ntrna_database_path=expand_path(_NTRNA_DATABASE_PATH.value),
                rfam_database_path=expand_path(_RFAM_DATABASE_PATH.value),
                rna_central_database_path=expand_path(_RNA_CENTRAL_DATABASE_PATH.value),
                pdb_database_path=expand_path(_PDB_DATABASE_PATH.value),
                seqres_database_path=expand_path(_SEQRES_DATABASE_PATH.value),
                jackhmmer_n_cpu=_JACKHMMER_N_CPU.value,
                nhmmer_n_cpu=_NHMMER_N_CPU.value,
                max_template_date=max_template_date,
            )
        else:
            data_pipeline_config = None

        if _RUN_INFERENCE.value:
            try:
                devices = jax.local_devices(backend='gpu')
                if not devices:
                    raise RuntimeError("No GPU devices found")
                
                print(f'Main process using GPU {_MAIN_GPU.value}')
                print(f'Will use GPU {_WORKER_GPU.value} for inference')
                
                # 检查GPU是否可用
                for gpu_id in [_MAIN_GPU.value, _WORKER_GPU.value]:
                    mem = get_available_gpu_memory(gpu_id)
                    print(f"GPU {gpu_id} available memory: {mem:.2f}GB")
                    if mem < 1:
                        print(f"Warning: GPU {gpu_id} has very low available memory!")
                
                # 检查模型目录
                model_path = pathlib.Path(MODEL_DIR.value)
                if not model_path.exists():
                    raise FileNotFoundError(f"Model directory not found: {model_path}")
                print(f"Using model directory: {model_path}")
                
                # 创建模型运行器
                print('Building model from scratch...')
                model_runner = ModelRunner(
                    config=make_model_config(
                        flash_attention_implementation=typing.cast(
                            attention.Implementation, _FLASH_ATTENTION_IMPLEMENTATION.value
                        ),
                        num_diffusion_samples=_NUM_DIFFUSION_SAMPLES.value,
                        num_recycles=_NUM_RECYCLES.value,
                        return_embeddings=_SAVE_EMBEDDINGS.value,
                    ),
                    device=devices[0],
                    model_dir=pathlib.Path(MODEL_DIR.value),
                )
                
                # 处理输入
                num_fold_inputs = 0
                for fold_input in fold_input_list:
                    try:
                        if _NUM_SEEDS.value is not None:
                            print(f'Expanding fold job {fold_input.name} to {_NUM_SEEDS.value} seeds')
                            fold_input = fold_input.with_multiple_seeds(_NUM_SEEDS.value)
                        process_fold_input(
                            fold_input=fold_input,
                            data_pipeline_config=data_pipeline_config,
                            model_runner=model_runner,
                            output_dir=os.path.join(_OUTPUT_DIR.value, fold_input.sanitised_name()),
                            buckets=tuple(int(bucket) for bucket in _BUCKETS.value),
                            conformer_max_iterations=_CONFORMER_MAX_ITERATIONS.value,
                            main_gpu=_MAIN_GPU.value,
                            worker_gpu=_WORKER_GPU.value,
                        )
                        num_fold_inputs += 1
                        print(f"Processed {num_fold_inputs}/{len(fold_input_list)} inputs")
                    except Exception as e:
                        print(f"Error processing input {fold_input.name}: {str(e)}")
                        raise

            except Exception as e:
                print(f"Fatal error during inference: {str(e)}")
                raise

        print(f'All jobs completed successfully')

    finally:
        # 确保清理预留进程
        gpu0_process.terminate()
        gpu1_process.terminate()
        gpu0_process.join()
        gpu1_process.join()


if __name__ == '__main__':
    flags.mark_flags_as_required(['output_dir'])
    app.run(main)
