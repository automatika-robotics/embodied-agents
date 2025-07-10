# Models / Vector Databases 🧠

Clients mentioned earlier take as input a **model** or **vector database (DB)** specification. These are in most cases generic wrappers around a class of models/dbs (e.g. transformers based LLMs) defined as [attrs](https://www.attrs.org/en/stable/) classes and include initialization parameters such as quantization schemes, inference options, embedding model (in case of vector DBs) etc. These specifications aim to standardize model initialization across diverse deployment platforms.

- 📚 [Available Models](../apidocs/agents/agents.models)
- 📚 [Available Vector DBs](../apidocs/agents/agents.vectordbs)

## Available Model Wrappers

```{list-table}
:widths: 20 80
:header-rows: 1
* - Model Name
  - Description

* - **[OllamaModel](../apidocs/agents/agents.models.md#classes)**
  - A LLM/MLLM model loaded from an Ollama checkpoint. Supports configurable generation and deployment options available in Ollama API. Complete list of Ollama models [here](https://ollama.com/library). This wrapper can be used with the OllamaClient.

* - **[TransformersLLM](../apidocs/agents/agents.models.md#classes)**
  - LLM models from HuggingFace/ModelScope based checkpoints. Supports quantization ("4bit", "8bit") specification. This model wrapper can be used with the GenericHTTPClient or any of the RoboML clients.

* - **[TransformersMLLM](../apidocs/agents/agents.models.md#classes)**
  - Multimodal LLM models from HuggingFace/ModelScope checkpoints for image-text inputs. Supports quantization. This model wrapper can be used with the GenericHTTPClient or any of the RoboML clients.

* - **[RoboBrain2](../apidocs/agents/agents.models.md#classes)**
  - [RoboBrain 2.0 by BAAI](https://github.com/FlagOpen/RoboBrain2.0) supports interactive reasoning with long-horizon planning and closed-loop feedback, spatial perception for precise point and bbox prediction from complex instructions and temporal perception for future trajectory estimation. Checkpoint defaults to `"BAAI/RoboBrain2.0-7B"`, with larger variants available [here](https://huggingface.co/collections/BAAI/robobrain20-6841eeb1df55c207a4ea0036). This wrapper can be used with any of the RoboML clients.

* - **[Whisper](../apidocs/agents/agents.models.md#classes)**
  - OpenAI's automatic speech recognition (ASR) model with various sizes (e.g., `"small"`, `"large-v3"`, etc.). These models are available on the [RoboML](https://github.com/automatika-robotics/roboml) platform and can be used with any RoboML client. Recommended, **RoboMLWSClient**.

* - **[SpeechT5](../apidocs/agents/agents.models.md#classes)**
  - Microsoft’s model for TTS synthesis. Configurable voice selection. This model is available on the [RoboML](https://github.com/automatika-robotics/roboml) platform and can be used with any RoboML client. Recommended, **RoboMLWSClient**.

* - **[Bark](../apidocs/agents/agents.models.md#classes)**
  - SunoAI’s Bark TTS model. Allows a selection [voices](https://suno-ai.notion.site/8b8e8749ed514b0cbf3f699013548683?v=bc67cff786b04b50b3ceb756fd05f68c). This model is available on the [RoboML](https://github.com/automatika-robotics/roboml) platform and can be used with any RoboML client. Recommended, **RoboMLWSClient**.

* - **[MeloTTS](../apidocs/agents/agents.models.md#classes)**
  - MyShell’s multilingual TTS model. Configure via `language` (e.g., `"JP"`) and `speaker_id` (e.g., `"JP-1"`). This model is available on the [RoboML](https://github.com/automatika-robotics/roboml) platform and can be used with any RoboML client. Recommended, **RoboMLWSClient**.

* - **[VisionModel](../apidocs/agents/agents.models.md#classes)**
  - A generic wrapper for object detection and tracking models available on [MMDetection framework](https://github.com/open-mmlab/mmdetection). Supports optional tracking, configurable thresholds, and deployment with TensorRT. This model is available on the [RoboML](https://github.com/automatika-robotics/roboml) platform and can be used with any RoboML client. Recommended, **RoboMLRESPClient**.
```

## Available Vector Databases

```{list-table}
:widths: 20 80
:header-rows: 1
* - Vector DB
  - Description

* - **[ChromaDB](../apidocs/agents/agents.vectordbs.md#classes)**
  - [Chroma](https://www.trychroma.com/) is an open-source AI application database with support for vector search, full-text search, and multi-modal retrieval. Supports "ollama" and "[sentence-transformers](https://sbert.net/)" embedding backends. Can be used with the ChomaClient.
```

````{note}
For `ChromaDB`, make sure you install required packages:

```bash
pip install ollama  # For Ollama backend (requires Ollama runtime)
pip install sentence-transformers  # For Sentence-Transformers backend
````

To use Ollama embedding models ([available models](https://ollama.com/search?c=embedding)), ensure the Ollama server is running and accessible via specified `host` and `port`.
