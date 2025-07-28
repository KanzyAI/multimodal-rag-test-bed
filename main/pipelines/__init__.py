import os
from main.embedders import EmbedderConfig, EmbedderFactory
from main.vector_dabases import DatabaseConfig, VectorDatabaseFactory

__all__ = ["database_mapping", "embedder_mapping", "TASK"]

TASK = os.getenv("DATASET").split('/')[-1].split('_')[-1].split('-')[-1].lower()

try:
    multimodal_single_database_config = DatabaseConfig(
        url=os.getenv("MULTIMODAL_SINGLE_DATABASE_URL"),
        api_key=os.getenv("MULTIMODAL_SINGLE_DATABASE_API_KEY"),
        collection_name=f"multimodal-single-{TASK}",
        vector_size=1024,
        vector_type="single",
    )
except Exception as e:
    multimodal_single_database_config = None

try:
    multimodal_multi_database_config = DatabaseConfig(
        url=os.getenv("MULTIMODAL_MULTI_DATABASE_URL"),
        api_key=os.getenv("MULTIMODAL_MULTI_DATABASE_API_KEY"),
        collection_name=f"multimodal-multi-{TASK}",
            vector_size=128,
            vector_type="multi",
        )
except Exception as e:
    multimodal_multi_database_config = None

try:
    text_multi_database_config = DatabaseConfig(
        url=os.getenv("TEXT_MULTI_DATABASE_URL"),
        api_key=os.getenv("TEXT_MULTI_DATABASE_API_KEY"),
        collection_name=f"text-multi-{TASK}",
            vector_size=128,
            vector_type="multi",
        )
except Exception as e:
    text_multi_database_config = None

try:
    text_single_database_config = DatabaseConfig(
        url=os.getenv("TEXT_SINGLE_DATABASE_URL"),
        api_key=os.getenv("TEXT_SINGLE_DATABASE_API_KEY"),
        collection_name=f"text-single-{TASK}",
        vector_size=128,
        vector_type="single",
    )
except Exception as e:
    text_single_database_config = None

database_mapping = {}

for key, value in os.environ.items():
    if key.startswith("MULTIMODAL_SINGLE_DATABASE_NAME"):
        database_mapping[key.split("_")[0] + "-" + key.split("_")[1]] = VectorDatabaseFactory.create_database(value, multimodal_single_database_config)
    elif key.startswith("MULTIMODAL_MULTI_DATABASE_NAME"):
        database_mapping[key.split("_")[0] + "-" + key.split("_")[1]] = VectorDatabaseFactory.create_database(value, multimodal_multi_database_config)
    elif key.startswith("TEXT_MULTI_DATABASE_NAME"):
        database_mapping[key.split("_")[0] + "-" + key.split("_")[1]] = VectorDatabaseFactory.create_database(value, text_multi_database_config)
    elif key.startswith("TEXT_SINGLE_DATABASE_NAME"):
        database_mapping[key.split("_")[0] + "-" + key.split("_")[1]] = VectorDatabaseFactory.create_database(value, text_single_database_config)

try:
    multimodal_single_embedder_config = EmbedderConfig(
        model_name=os.getenv("MULTIMODAL_SINGLE_EMBEDDER"),
    )
except Exception as e:
    multimodal_single_embedder_config = None

try:
    multimodal_multi_embedder_config = EmbedderConfig(
        model_name=os.getenv("MULTIMODAL_MULTI_EMBEDDER"),
    )
except Exception as e:
    multimodal_multi_embedder_config = None

try:
    text_multi_embedder_config = EmbedderConfig(
        model_name=os.getenv("TEXT_MULTI_EMBEDDER"),
    )
except Exception as e:
    text_multi_embedder_config = None

try:
    text_single_embedder_config = EmbedderConfig(
        model_name=os.getenv("TEXT_SINGLE_EMBEDDER"),
    )
except Exception as e:
    text_single_embedder_config = None

embedder_mapping = {}

for key, value in os.environ.items():
    if key.startswith("MULTIMODAL_SINGLE_EMBEDDER"):
        key = key.split("_")[0] + "-" + key.split("_")[1]
        embedder_mapping[key] = EmbedderFactory.create_embedder(value, multimodal_single_embedder_config)
    elif key.startswith("MULTIMODAL_MULTI_EMBEDDER"):
        key = key.split("_")[0] + "-" + key.split("_")[1]
        embedder_mapping[key] = EmbedderFactory.create_embedder(value, multimodal_multi_embedder_config)
    elif key.startswith("TEXT_MULTI_EMBEDDER"):
        key = key.split("_")[0] + "-" + key.split("_")[1]
        embedder_mapping[key] = EmbedderFactory.create_embedder(value, text_multi_embedder_config)
    elif key.startswith("TEXT_SINGLE_EMBEDDER"):
        key = key.split("_")[0] + "-" + key.split("_")[1]
        embedder_mapping[key] = EmbedderFactory.create_embedder(value, text_single_embedder_config)