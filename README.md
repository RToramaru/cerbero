# Descrição

O cerbero é um sistema de detecção de placas veiculares.
Desenvolvido como projeto de pesquisa, o cerbero foi pensado para melhorar o funcionamento do IFNMG - Campus Salinas, realizando o registro automático de placas veiculares.

#### Implementação

O cerbero foi implementado em duas vias, a parte de processamento, desenvolvido em Python 3.1 e a parte Web para visualização, desenvolvido em PHP 8.2.
A aplicação Python realiza a detecção das placas, sua obtenção é realizada atráves de três modelos de rede neural treinado, uma para detecção das placa, uma para a detecção da reagião dos caractres, e outra para realizar o ocr. Esses três modelos foram treinados utilizando o YOLOv4 com a base de dados Sense-ALPR. 

Link para o site do YOLOv4
https://pjreddie.com/darknet/yolo/

Link para o site da base de dados Sense-ALPR
http://smartsenselab.dcc.ufmg.br/dataset/banco-de-dados-sense-alpr/


#### Resultados

Para critério de resultados foram realizados duas validações, uma com a base de dados para validação, e outro com os dados reais obtidos no Campus - Salinas.

Na base de dados os resultados obtidos foram:

| ACURÁCIA |  PRECISÃO | REVOCAÇÃO |  F1-SCORE |
| ------ | ------ | ------ | ------ |
| 0,9344 |  0,9325 |  0,9920 |  0,9613 |


Já utilizando os dados reais obtidos no Campus os resultados foram divididos em três partes, detecção da placa, da reagião dos caracteres, e o ocr:

|Parte | ACURÁCIA |  PRECISÃO | REVOCAÇÃO |  F1-SCORE |
| ------ | ------ | ------ | ------ |------ |
|Placa | 0,9568 |  0,9347 | 0,9932 |  0,9630 |
|Região | 1 |  1 | 1 |  1 |
|Caracteres | 0,7541 |  0,7332 | 0,7145 |  0,7237 |

### Instalação
Para realizar a instalação do projeto realize as etapas descritas a baixo: 

- Clone o repositório 
- Instale as dependências do projeto com o comando
```sh
$ pip3 install -r requirements.txt
```

### Utilização

Para se utilizar do sistema basta executar o arquivo cerbero.py.
O arquivo pode ser executado de duas maneiras, a primeira através de um video ja pronto, e na segunda utilizando-se da imagem diretamente da câmera, essas utilizações estão escritas no código.

# Requisitos necessários

Python 3.*
PHP 8+

### Demonstração

![](/screens/1.png)
![](/screens/2.png)