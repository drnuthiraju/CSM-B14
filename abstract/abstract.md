# Project Abstract

## AI Hallucination Detection and Verification System using Large Language Models, Retrieval-Augmented Generation, and Multi-Agent AI

Large Language Models (LLMs) such as GPT, LLaMA, and Mistral have demonstrated strong performance in natural language understanding and generation. However, these models can produce hallucinated information—statements that are factually incorrect, unsupported by available evidence, or contradictory to the provided context. This creates reliability concerns when LLMs are used in applications where accuracy and trustworthiness are important.

Retrieval-Augmented Generation (RAG) improves the reliability of LLM responses by providing relevant external information as context. However, research has shown that hallucinations can still occur even when relevant retrieved information is available. The RAGTruth dataset demonstrates that LLM-generated responses can contain unsupported or contradictory claims despite the presence of retrieval context.

This project proposes an AI-based hallucination detection and verification system that analyzes LLM-generated responses and evaluates their consistency with available evidence. The proposed system will combine Retrieval-Augmented Generation, hallucination detection techniques, and a multi-agent verification architecture. Different agents will perform tasks such as information retrieval, response analysis, factual verification, and evidence comparison.

The system will identify potentially hallucinated portions of a response and provide a confidence or reliability assessment along with supporting evidence. The project will be evaluated using standard metrics such as precision, recall, and F1-score for hallucination detection. Where applicable, evaluation will also consider detection at the response level and the specific span or portion of text containing the hallucination.

The overall objective is to develop a practical and explainable framework that can improve the reliability and trustworthiness of LLM-generated responses.