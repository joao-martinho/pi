# Diagrama de atividades

Abaixo está o diagrama de atividades (no estilo _flowchart_) que representa a nossa solução do trabalho, renderizado pela ferramenta Mermaid.

```mermaid
flowchart TD
    A([Início]) --> B["Preparar ambiente<br/>Importar libs e imagens"]
    B --> C["Isolar célula<br/>Threshold simples"]
    C --> D["Segmentação inicial<br/>Tons de cinza"]
    D --> E{Suficiente?}
    E -->|Sim| F["Mostrar resultado<br/>Segmentação parcial"]
    E -->|Não| G["Ajustar parâmetros HSV<br/>Trackbars interativas"]
    G --> H["Gerar mapa de probabilidade<br/>Cor, saturação e intensidade"]
    H --> I["Refinar com bordas<br/>Sobel + anti-borda"]
    I --> J["Aplicar Watershed<br/>Marcadores + separação"]
    J --> K["Processar em lote<br/>Todas as imagens"]
    K --> L([Fim])
```