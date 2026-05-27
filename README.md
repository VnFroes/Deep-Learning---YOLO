# Automated PPE Safety Auditing using Computer Vision

**Live Demo:** [Hugging Face Space - Automated PPE Audit](https://huggingface.co/spaces/EmLL00000/Automated_audit_through_detection_of_personal_protective_equipment_PPE_using_YOLO)

**Dataset used:** [PPE Dataset](https://www.kaggle.com/datasets/waquarahmed1/ppe-dataset)

## Project Overview
This project implements a Proof of Concept (POC) for an Automated Auditing and Compliance Analysis system tailored for the construction industry. Rather than focusing on real-time video monitoring—which requires heavy infrastructure—this tool is designed to analyze static visual records of past events. The objective is to identify the presence or absence of Personal Protective Equipment (PPE) to support corrective actions, safety feedback, and targeted worker training.

The system detects five classes of PPE: **Helmets, Masks, Safety Vests, Boots, and Gloves**.

## Technical Architecture
*   **Core Model:** YOLO26n (Nano architecture)
*   **Frameworks:** PyTorch, Ultralytics
*   **Image Processing:** OpenCV, NumPy
*   **Deployment:** Gradio, Hugging Face Spaces

## Development Pipeline & Engineering Choices

### 1. Exploratory Data Analysis (EDA) & The Small Object Problem
Initial validation tests indicated lower performance for "Gloves" and "Boots". A common assumption in object detection is class imbalance. However, custom EDA scripts revealed a highly balanced dataset (e.g., 5,847 gloves vs. 4,578 safety vests). 
The bottleneck was identified not as a lack of data, but as a physical image constraint: gloves and boots are physically small (few pixels at 640x640 resolution) and suffer from severe occlusion in construction environments.

### 2. Hardware Optimization & I/O Management
Training on cloud environments (Google Colab / Nvidia Tesla T4) presented an I/O bottleneck. Reading 13,000 images per epoch from a cloud drive starved the GPU, resulting in excessive training times.
*   **Solution:** Implemented `cache='disk'` to convert and store images as highly optimized binary files on the local SSD, bypassing network latency.
*   **VRAM Maximization:** Increased the batch size to `batch=64`, fully utilizing the 16GB VRAM of the T4 GPU. This reduced the training time drastically, allowing for a healthy 50-epoch training cycle without hardware timeouts.

### 3. Hyperparameter Engineering (Data Augmentation)
To solve the small object and occlusion challenges without artificially increasing the image resolution (which would waste compute resources), specific hyperparameters were engineered:
*   **Optimizer:** Transitioned to `AdamW` for faster and more stable convergence within the 50-epoch limit.
*   **Mosaic & Scale:** Aggressively increased Mosaic augmentation. By stitching four images together during training, the relative size of objects was reduced, forcing the neural network to specialize in detecting tiny, occluded features (like a dark glove holding a tool).

### 4. Deployment & UI/UX
The final model weights (`best.pt`) were deployed to Hugging Face Spaces using a custom Gradio interface.
*   **Color Channel Correction:** Addressed the standard computer vision discrepancy where web interfaces provide RGB arrays while OpenCV/YOLO expects BGR. Explicit conversion layers were coded to prevent inference failure.
*   **Stateless Auditing:** The application processes the image, generates a dynamic Markdown compliance report based on the detected tensor classes, and discards the data, ensuring a lightweight and privacy-compliant architecture.

## Performance Metrics
The final model achieved robust convergence with zero signs of overfitting, yielding the following metrics on the validation set:
*   **mAP50 (Mean Average Precision):** 0.865
*   **Precision:** 0.881
*   **Recall:** 0.798

---
---

# Auditoria Automatizada de EPIs utilizando Visão Computacional

**Demonstração Online:** [Hugging Face Space - Automated PPE Audit](https://huggingface.co/spaces/EmLL00000/Automated_audit_through_detection_of_personal_protective_equipment_PPE_using_YOLO)

**Dataset usado:** [PPE Dataset](https://www.kaggle.com/datasets/waquarahmed1/ppe-dataset)

## Visão Geral do Projeto
Este projeto implementa uma Prova de Conceito (POC) para um sistema de Auditoria Automatizada e Análise de Conformidade voltado para a construção civil. Em vez de focar em monitoramento de vídeo em tempo real — o que exige infraestrutura pesada —, esta ferramenta foi projetada para analisar registros visuais estáticos de eventos passados. O objetivo é identificar a presença ou ausência de Equipamentos de Proteção Individual (EPIs) para embasar ações corretivas, feedbacks de segurança e treinamentos direcionados.

O sistema detecta cinco classes de EPIs: **Capacetes, Máscaras, Coletes de Segurança, Botas e Luvas**.

## Arquitetura Técnica
*   **Modelo Principal:** YOLO26n (Arquitetura Nano)
*   **Frameworks:** PyTorch, Ultralytics
*   **Processamento de Imagem:** OpenCV, NumPy
*   **Deploy:** Gradio, Hugging Face Spaces

## Pipeline de Desenvolvimento e Decisões de Engenharia

### 1. Análise Exploratória de Dados (EDA) e o Problema de Objetos Pequenos
Testes de validação iniciais indicaram desempenho inferior para "Luvas" e "Botas". Uma suposição comum em detecção de objetos é o desbalanceamento de classes. No entanto, scripts de EDA revelaram um dataset altamente balanceado (ex: 5.847 luvas contra 4.578 coletes).
O gargalo foi identificado não como falta de dados, mas como uma restrição física da imagem: luvas e botas são fisicamente pequenas (poucos pixels na resolução 640x640) e sofrem oclusão severa em ambientes de obra.

### 2. Otimização de Hardware e Gerenciamento de I/O
O treinamento em ambientes de nuvem (Google Colab / Nvidia Tesla T4) apresentou um gargalo de I/O (Entrada/Saída). Ler 13.000 imagens por época de um disco em nuvem deixava a GPU ociosa, resultando em tempos de treinamento excessivos.
*   **Solução:** Implementação do parâmetro `cache='disk'` para converter e armazenar as imagens como arquivos binários altamente otimizados no SSD local, contornando a latência de rede.
*   **Maximização da VRAM:** Aumento do tamanho do lote para `batch=64`, utilizando integralmente os 16GB de VRAM da GPU T4. Isso reduziu o tempo de treinamento drasticamente, permitindo um ciclo saudável de 50 épocas sem interrupções do servidor.

### 3. Engenharia de Hiperparâmetros (Data Augmentation)
Para resolver os desafios de objetos pequenos e oclusão sem aumentar artificialmente a resolução da imagem (o que desperdiçaria recursos computacionais), hiperparâmetros específicos foram ajustados:
*   **Otimizador:** Transição para `AdamW` visando uma convergência mais rápida e estável dentro do limite de 50 épocas.
*   **Mosaic e Scale:** Aumento agressivo do aumento de dados via Mosaic. Ao unir quatro imagens durante o treinamento, o tamanho relativo dos objetos foi reduzido, forçando a rede neural a se especializar na detecção de características minúsculas e ocluídas (como uma luva escura segurando uma ferramenta).

### 4. Deploy e UI/UX
Os pesos finais do modelo (`best.pt`) foram implantados no Hugging Face Spaces utilizando uma interface customizada em Gradio.
*   **Correção de Canal de Cor:** Solução da discrepância padrão em visão computacional onde interfaces web fornecem matrizes RGB enquanto OpenCV/YOLO espera BGR. Camadas de conversão explícitas foram codificadas para evitar falhas na inferência.
*   **Auditoria Stateless:** A aplicação processa a imagem, gera um relatório de conformidade dinâmico em Markdown baseado nas classes do tensor detectado e descarta os dados, garantindo uma arquitetura leve e em conformidade com diretrizes de privacidade.

## Métricas de Desempenho
O modelo final alcançou uma convergência robusta com zero sinais de sobreajuste (overfitting), obtendo as seguintes métricas no conjunto de validação:
*   **mAP50 (Mean Average Precision):** 0.865
*   **Precisão:** 0.881
*   **Revocação (Recall):** 0.798
