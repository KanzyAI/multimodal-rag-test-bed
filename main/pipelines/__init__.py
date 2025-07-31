import os
from main.embedders import EmbedderConfig, EmbedderFactory
from main.vector_dabases import DatabaseConfig, VectorDatabaseFactory

__all__ = ["database_mapping", "embedder_mapping", "TASK"]

TASK = os.getenv("TASK")

PIPELINE_NAME = os.getenv("PIPELINE_NAME")

database_mapping = {}

multimodal_single_database_config = DatabaseConfig(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY"),
    collection_name=f"multimodal-single-{TASK}",
    vector_size=3584,
    vector_type="single",
)

multimodal_multi_database_config = DatabaseConfig(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY"),
    collection_name=f"multimodal-multi-{TASK}",
        vector_size=128,
        vector_type="multi",
    )

text_single_database_config = DatabaseConfig(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY"),
    collection_name=f"text-single-{TASK}",
    vector_size=4096,
    vector_type="single",
)

text_multi_database_config = DatabaseConfig(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY"),
    collection_name=f"text-multi-{TASK}",
        vector_size=128,
        vector_type="multi",
    )

if PIPELINE_NAME == "MULTIMODAL-SINGLE":
    database_mapping["multimodal-single"] = VectorDatabaseFactory.create_database("qdrant",multimodal_single_database_config)
elif PIPELINE_NAME == "MULTIMODAL-MULTI":
    database_mapping["multimodal-multi"] = VectorDatabaseFactory.create_database("qdrant",multimodal_multi_database_config)
elif PIPELINE_NAME == "TEXT-MULTI":
    database_mapping["text-multi"] = VectorDatabaseFactory.create_database("qdrant",text_multi_database_config)
elif PIPELINE_NAME == "TEXT-SINGLE":
    database_mapping["text-single"] = VectorDatabaseFactory.create_database("qdrant",text_single_database_config)
else:
    for pipeline in ["multimodal-single", "multimodal-multi", "text-multi", "text-single"]:
        database_mapping[pipeline] = VectorDatabaseFactory.create_database("qdrant",globals()[f"{pipeline}_database_config"])


if PIPELINE_NAME == "MULTIMODAL-SINGLE":
    multimodal_single_embedder_config = EmbedderConfig(
        model_name=os.getenv("MULTIMODAL_SINGLE_EMBEDDER"),
    )

if PIPELINE_NAME == "MULTIMODAL-MULTI":
    multimodal_multi_embedder_config = EmbedderConfig(
        model_name=os.getenv("MULTIMODAL_MULTI_EMBEDDER"),
    )

if PIPELINE_NAME == "TEXT-MULTI":
    text_multi_embedder_config = EmbedderConfig(
        model_name=os.getenv("TEXT_MULTI_EMBEDDER"),
    )

if PIPELINE_NAME == "TEXT-SINGLE":
    text_single_embedder_config = EmbedderConfig(
        model_name=os.getenv("TEXT_SINGLE_EMBEDDER"),
    )

embedder_mapping = {}

if PIPELINE_NAME == "MULTIMODAL-SINGLE":
    embedder_mapping["multimodal-single"] = EmbedderFactory.create_embedder(os.getenv("MULTIMODAL_SINGLE_EMBEDDER"), multimodal_single_embedder_config)
elif PIPELINE_NAME == "MULTIMODAL-MULTI":
    embedder_mapping["multimodal-multi"] = EmbedderFactory.create_embedder(os.getenv("MULTIMODAL_MULTI_EMBEDDER"), multimodal_multi_embedder_config)
elif PIPELINE_NAME == "TEXT-MULTI":
    embedder_mapping["text-multi"] = EmbedderFactory.create_embedder(os.getenv("TEXT_MULTI_EMBEDDER"), text_multi_embedder_config)
elif PIPELINE_NAME == "TEXT-SINGLE":
    embedder_mapping["text-single"] = EmbedderFactory.create_embedder(os.getenv("TEXT_SINGLE_EMBEDDER"), text_single_embedder_config)
else:
    for pipeline in ["multimodal-single", "multimodal-multi", "text-multi", "text-single"]:
        embedder_mapping[pipeline] = EmbedderFactory.create_embedder(os.getenv(f"{pipeline.upper()}_EMBEDDER"), globals()[f"{pipeline}_embedder_config"])
