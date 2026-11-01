"""Check the serving interfaces required by generation and v4 mining."""
import importlib.metadata

from ray.data.llm import ServeDeploymentProcessorConfig, build_processor
from ray.llm._internal.serve.core.configs.openai_api_models import EmbeddingCompletionRequest, ScoreRequest
from ray.serve.llm import LLMConfig, ModelLoadingConfig, build_llm_deployment
from ray.serve.llm.openai_api_models import ChatCompletionRequest
from vllm import LLM

for name in ('ray', 'vllm', 'torch', 'transformers', 'usearch'):
    print(f'{name}=={importlib.metadata.version(name)}')
print('Required interfaces import successfully; next run a small GPU shard.')
