# TCC-UTFPR

Meu trabalho de conclusão do curso de Engenharia de Software — repositório com código, dados e o relatório final do projeto.

## Descrição do projeto

Este repositório contém o trabalho de conclusão de curso do autor, incluindo o relatório em PDF, o código-fonte em Python e conjuntos de dados usados para treino e teste. Os artefatos servem para reproduzir os experimentos descritos no trabalho e para estudar a implementação empregada no projeto.

## Conteúdo do repositório

- **Trabalho_de_conclusão_de_curso.pdf** — relatório final do TCC.
- **Código_Fonte.py** — script principal em Python com a lógica do projeto.
- **TabelaTreino.csv** — conjunto de dados utilizado para treino.
- **TabelaTeste3M.CSV** — conjunto de dados de teste.
- **Tabela3MOriginal.xlsx** — versão original dos dados em Excel.
- **.gitattributes** — configurações Git do repositório.

## Requisitos

- Python 3.8+ recomendado.
- Bibliotecas Python possivelmente utilizadas (instale conforme necessidade do script): `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`.
- Ferramentas para abrir o relatório: leitor de PDF; para o Excel, software compatível (LibreOffice, Excel etc.).

## Como executar

1.  Clone o repositório.
2.  Crie um ambiente virtual e instale dependências:
    ```bash
    python -m venv venv
    source venv/bin/activate  # Linux/macOS
    # ou
    venv\Scripts\activate  # Windows
    ```
    ```bash
    pip install pandas numpy scikit-learn matplotlib seaborn
    ```
3.  Ajuste caminhos para os arquivos de dados dentro de `Código_Fonte.py` se necessário.
4.  Execute o script principal:
    ```bash
    python Código_Fonte.py
    ```

**Observação sobre execução:** verifique no início do arquivo `Código_Fonte.py` se há parâmetros configuráveis (nomes de arquivos, opções de pré-processamento ou de saída) e atualize conforme sua máquina antes de rodar.

## Estrutura e pontos de atenção

-   Os arquivos CSV/XLSX contêm os dados usados no experimento; verifique a codificação e separadores ao carregar com pandas.
-   O script pode gerar saídas (gráficos, modelos treinados, métricas) no diretório atual — revise o código para confirmar onde os arquivos são escritos.
-   Se encontrar erros por versão de bibliotecas, tente atualizar ou ajustar chamadas de API conforme a versão instalada.

## Uso dos dados

-   **TabelaTreino.csv:** use para treinar modelos e replicar os experimentos do relatório.
-   **TabelaTeste3M.CSV** usada para validação e análises comparativas.
-   Trate dados sensíveis conforme apropriado antes de compartilhar.

## Referências

O relatório completo e os detalhes metodológicos estão no arquivo PDF do repositório.

## Contato

-   Autor do repositório: [pedrovinchi1](https://github.com/pedrovinchi1) (GitHub).
-   Para dúvidas sobre o TCC
