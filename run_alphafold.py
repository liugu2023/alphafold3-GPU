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
from typing import overload
import hashlib

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

# 添加新的配置选项
_JAX_CACHE_SIZE = flags.DEFINE_integer(
    'jax_cache_size',
    5000,  # 默认缓存5000个编译结果
    '设置JAX编译缓存大小(MB)',
)

_JAX_CACHE_EVICTION = flags.DEFINE_enum(
    'jax_cache_eviction',
    default='lru',
    enum_values=['lru', 'fifo'],
    help='JAX缓存淘汰策略'
)

# 添加新的配置选项
_FEATURE_CACHE_DIR = flags.DEFINE_string(
    'feature_cache_dir',
    None,
    '特征缓存目录路径'
)

_NUM_FEATURE_WORKERS = flags.DEFINE_integer(
    'num_feature_workers',
    None,
    '特征化并行处理的worker数量'
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

  def extract_inference_results_and_maybe_embeddings(
      self,
      batch: features.BatchDict,
      result: model.ModelResult,
      target_name: str,
  ) -> tuple[list[model.InferenceResult], dict[str, np.ndarray] | None]:
    """Extracts inference results and embeddings (if set) from model outputs."""
    inference_results = list(
        model.Model.get_inference_result(
            batch=batch, result=result, target_name=target_name
        )
    )
    num_tokens = len(inference_results[0].metadata['token_chain_ids'])
    embeddings = {}
    if 'single_embeddings' in result:
      embeddings['single_embeddings'] = result['single_embeddings'][:num_tokens]
    if 'pair_embeddings' in result:
      embeddings['pair_embeddings'] = result['pair_embeddings'][
          :num_tokens, :num_tokens
      ]
    return inference_results, embeddings or None


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
    buckets: Sequence[int] | None = None,
    conformer_max_iterations: int | None = None,
) -> Sequence[ResultsForSeed]:
  """Runs the full inference pipeline to predict structures for each seed."""

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
      f'Featurising data with {len(fold_input.rng_seeds)} seed(s) took'
      f' {time.time() - featurisation_start_time:.2f} seconds.'
  )
  print(
      'Running model inference and extracting output structure samples with'
      f' {len(fold_input.rng_seeds)} seed(s)...'
  )
  all_inference_start_time = time.time()
  all_inference_results = []
  for seed, example in zip(fold_input.rng_seeds, featurised_examples):
    print(f'Running model inference with seed {seed}...')
    inference_start_time = time.time()
    rng_key = jax.random.PRNGKey(seed)
    result = model_runner.run_inference(example, rng_key)
    print(
        f'Running model inference with seed {seed} took'
        f' {time.time() - inference_start_time:.2f} seconds.'
    )
    print(f'Extracting inference results with seed {seed}...')
    extract_structures = time.time()
    inference_results, embeddings = (
        model_runner.extract_inference_results_and_maybe_embeddings(
            batch=example, result=result, target_name=fold_input.name
        )
    )
    print(
        f'Extracting {len(inference_results)} inference samples with'
        f' seed {seed} took {time.time() - extract_structures:.2f} seconds.'
    )

    all_inference_results.append(
        ResultsForSeed(
            seed=seed,
            inference_results=inference_results,
            full_fold_input=fold_input,
            embeddings=embeddings,
        )
    )
  print(
      'Running model inference and extracting output structures with'
      f' {len(fold_input.rng_seeds)} seed(s) took'
      f' {time.time() - all_inference_start_time:.2f} seconds.'
  )
  return all_inference_results


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
) -> folding_input.Input | Sequence[ResultsForSeed]:
    """处理单个fold输入"""
    print(f'\nRunning fold job {fold_input.name}...')
    
    # 初始化特征缓存
    feature_cache = FeatureCache(_FEATURE_CACHE_DIR.value)
    
    # 尝试从缓存加载特征
    cached_features = feature_cache.get_cached_features(fold_input)
    if cached_features is not None:
        print('Using cached features')
        featurised_examples = cached_features
    else:
        print(f'Featurising data with {len(fold_input.rng_seeds)} seed(s)...')
        featurisation_start_time = time.time()
        
        # 获取CCD实例
        ccd = chemical_components.cached_ccd(user_ccd=fold_input.user_ccd)
        
        # 并行特征化处理
        featurised_examples = parallel_featurisation(
            fold_input=fold_input,
            buckets=buckets,
            ccd=ccd,
            num_workers=_NUM_FEATURE_WORKERS.value
        )
        
        print(
            f'Featurising data took {time.time() - featurisation_start_time:.2f}'
            ' seconds.'
        )
        
        # 缓存特征
        feature_cache.cache_features(fold_input, featurised_examples)

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


def configure_jax_cache():
    """配置JAX编译缓存"""
    if _JAX_COMPILATION_CACHE_DIR.value is not None:
        # 设置缓存目录
        jax.config.update('jax_compilation_cache_dir', _JAX_COMPILATION_CACHE_DIR.value)
        
        # 设置缓存大小
        jax.config.update('jax_compilation_cache_size_mb', _JAX_CACHE_SIZE.value)
        
        # 设置缓存淘汰策略
        jax.config.update('jax_compilation_cache_eviction_policy', _JAX_CACHE_EVICTION.value)
        
        print(f'JAX compilation cache configured:')
        print(f'- Cache directory: {_JAX_COMPILATION_CACHE_DIR.value}')
        print(f'- Cache size: {_JAX_CACHE_SIZE.value}MB')
        print(f'- Eviction policy: {_JAX_CACHE_EVICTION.value}')


def optimize_bucket_sizes(fold_inputs: Sequence[folding_input.Input]) -> list[int]:
    """根据输入序列长度分布优化bucket大小"""
    # 收集所有序列长度
    seq_lengths = []
    for fold_input in fold_inputs:
        for chain in fold_input.chains:
            seq_lengths.append(len(chain.sequence))
    
    # 计算序列长度分布
    seq_lengths = np.array(seq_lengths)
    percentiles = np.percentile(seq_lengths, [25, 50, 75, 90, 95, 99])
    
    # 生成优化后的bucket大小
    buckets = []
    for p in percentiles:
        # 向上取整到最近的64的倍数
        bucket_size = int(np.ceil(p / 64) * 64)
        if bucket_size not in buckets:
            buckets.append(bucket_size)
    
    # 确保有足够大的bucket处理极端情况
    max_length = max(seq_lengths)
    if max_length > buckets[-1]:
        buckets.append(int(np.ceil(max_length / 64) * 64))
        
    return sorted(buckets)

def update_buckets(fold_inputs: Sequence[folding_input.Input]):
    """更新bucket配置"""
    if not _BUCKETS.value:  # 如果未指定bucket sizes
        optimized_buckets = optimize_bucket_sizes(fold_inputs)
        flags.FLAGS.buckets = [str(b) for b in optimized_buckets]
        print(f'Optimized bucket sizes: {optimized_buckets}')


def warmup_compilation(
    model_runner: ModelRunner,
    example_inputs: Sequence[features.BatchDict]
) -> None:
    """预热JAX编译缓存"""
    print('Warming up JAX compilation cache...')
    start_time = time.time()
    
    # 对每个bucket size进行预热
    for bucket_size in map(int, _BUCKETS.value):
        # 创建示例输入
        dummy_input = create_dummy_batch(bucket_size)
        # 运行一次前向传播进行预热
        _ = model_runner.run_inference(
            dummy_input,
            jax.random.PRNGKey(0)
        )
        
    print(f'Compilation warmup took {time.time() - start_time:.2f} seconds')

def create_dummy_batch(size: int) -> features.BatchDict:
    """创建指定大小的示例输入用于预热"""
    # 创建符合模型输入要求的dummy数据
    return {
        'aatype': jnp.zeros((size,), dtype=jnp.int32),
        'residue_index': jnp.arange(size),
        # ... 其他必要的特征字段
    }

def parallel_featurisation(
    fold_input: folding_input.Input,
    buckets: Sequence[int] | None,
    ccd: chemical_components.ChemicalComponentDictionary,
    num_workers: int | None = None
) -> list[features.BatchDict]:
    """并行处理特征化"""
    if num_workers is None:
        num_workers = min(len(fold_input.rng_seeds), multiprocessing.cpu_count())
    
    print(f'Running parallel featurisation with {num_workers} workers')
    
    with multiprocessing.Pool(num_workers) as pool:
        # 将输入分成多个批次
        featurisation_args = [
            (fold_input, seed, buckets, ccd)
            for seed in fold_input.rng_seeds
        ]
        
        # 并行处理每个批次
        results = pool.starmap(
            featurise_single_input,
            featurisation_args
        )
        
    return results

def featurise_single_input(
    fold_input: folding_input.Input,
    seed: int,
    buckets: Sequence[int] | None,
    ccd: chemical_components.ChemicalComponentDictionary,
) -> features.BatchDict:
    """处理单个输入的特征化"""
    # 创建单个seed的fold input
    single_seed_input = fold_input.with_single_seed(seed)
    return featurisation.featurise_input(
        fold_input=single_seed_input,
        buckets=buckets,
        ccd=ccd,
        verbose=False  # 避免并行时的输出混乱
    )

class FeatureCache:
    """特征缓存管理"""
    def __init__(self, cache_dir: str | None = None):
        self.cache_dir = cache_dir
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
    
    def get_cache_key(self, fold_input: folding_input.Input) -> str:
        """生成缓存键"""
        # 使用输入的关键信息生成唯一标识
        key_components = [
            fold_input.name,
            *(chain.sequence for chain in fold_input.chains),
            str(fold_input.rng_seeds)
        ]
        return hashlib.md5('_'.join(key_components).encode()).hexdigest()
    
    def get_cached_features(
        self, 
        fold_input: folding_input.Input
    ) -> list[features.BatchDict] | None:
        """获取缓存的特征"""
        if not self.cache_dir:
            return None
            
        cache_key = self.get_cache_key(fold_input)
        cache_path = os.path.join(self.cache_dir, f'{cache_key}.npz')
        
        if os.path.exists(cache_path):
            try:
                with np.load(cache_path, allow_pickle=True) as data:
                    return data['features']
            except Exception as e:
                print(f'Failed to load cache: {e}')
                return None
        return None
    
    def cache_features(
        self,
        fold_input: folding_input.Input,
        features_list: list[features.BatchDict]
    ) -> None:
        """缓存特征"""
        if not self.cache_dir:
            return
            
        cache_key = self.get_cache_key(fold_input)
        cache_path = os.path.join(self.cache_dir, f'{cache_key}.npz')
        
        try:
            np.savez_compressed(
                cache_path,
                features=features_list
            )
        except Exception as e:
            print(f'Failed to save cache: {e}')

def optimize_feature_preprocessing(
    fold_input: folding_input.Input,
    buckets: Sequence[int] | None
) -> features.BatchDict:
    """优化特征预处理"""
    # 1. 提前分配内存
    max_length = max(len(chain.sequence) for chain in fold_input.chains)
    if buckets:
        max_length = min(max(buckets), max_length)
    
    # 2. 预分配特征数组
    prealloc_features = {
        'aatype': np.zeros((max_length,), dtype=np.int32),
        'residue_index': np.arange(max_length),
        'seq_length': np.array([max_length], dtype=np.int32),
        # ... 其他特征
    }
    
    # 3. 使用向量化操作
    for chain in fold_input.chains:
        seq_length = len(chain.sequence)
        prealloc_features['aatype'][:seq_length] = np.array([
            chemical_components.residue_to_index(aa)
            for aa in chain.sequence
        ])
    
    return prealloc_features

def main(_):
    # 配置JAX缓存
    configure_jax_cache()
    
    # 加载输入数据
    if _INPUT_DIR.value is not None:
        fold_inputs = folding_input.load_fold_inputs_from_dir(
            pathlib.Path(_INPUT_DIR.value)
        )
    elif _JSON_PATH.value is not None:
        fold_inputs = folding_input.load_fold_inputs_from_path(
            pathlib.Path(_JSON_PATH.value)
        )
    
    # 优化bucket sizes
    update_buckets(fold_inputs)
    
    if _RUN_INFERENCE.value:
        # 初始化模型
        model_runner = ModelRunner(
            config=make_model_config(
                flash_attention_implementation=typing.cast(
                    attention.Implementation, _FLASH_ATTENTION_IMPLEMENTATION.value
                ),
                num_diffusion_samples=_NUM_DIFFUSION_SAMPLES.value,
                num_recycles=_NUM_RECYCLES.value,
                return_embeddings=_SAVE_EMBEDDINGS.value,
            ),
            device=jax.local_devices(backend='gpu')[_GPU_DEVICE.value],
            model_dir=pathlib.Path(MODEL_DIR.value),
        )
        
        # 创建示例输入进行编译预热
        example_inputs = [
            create_dummy_batch(int(bucket))
            for bucket in _BUCKETS.value[:3]  # 只预热前几个常用size
        ]
        warmup_compilation(model_runner, example_inputs)
        
        # 继续处理实际输入...

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
        devices = jax.local_devices(backend='gpu')
        print(
            f'Found local devices: {devices}, using device {_GPU_DEVICE.value}:'
            f' {devices[_GPU_DEVICE.value]}'
        )

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
            device=devices[_GPU_DEVICE.value],
            model_dir=pathlib.Path(MODEL_DIR.value),
        )
        # Check we can load the model parameters before launching anything.
        print('Checking that model parameters can be loaded...')
        _ = model_runner.model_params
    else:
        model_runner = None

    num_fold_inputs = 0
    for fold_input in fold_inputs:
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
        )
        num_fold_inputs += 1

    print(f'Done running {num_fold_inputs} fold jobs.')


if __name__ == '__main__':
  flags.mark_flags_as_required(['output_dir'])
  app.run(main)