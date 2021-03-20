# lap1
# DISCIPLINA: Laboratório de Pesquisa 1 - lap1

# TRABALHO 01:  Título do Trabalho
Trabalho desenvolvido durante a disciplina:

# Sumário

### 1. Componentes <br>
Integrantes do grupo<br>
Caicke Pinheiro: caicke@gmail.com<br>
Eduardo Alves Figueiredo: eduardomatanob@gmail.com<br>
Hellesandro Gonzaga de Carvalho: hellesandro@hotmail.com<br>

### 2. Apresentação dos Datasets (Clássico + Em estudo)<br>

Link do colab da base de dados clássica: https://colab.research.google.com/drive/1yFG-3XkXxVKyuE63siBg9sbx0sAawBf4#scrollTo=VrZN-PDgFXO7 

>#### 2.1 Visão geral da base de dados clássica:<br>
 * **P:** Seus dados são sobre o que?
    > **R:** Base de dados com informações sobre os tripulantes do Titanic.

* **P:** O que você deseja com este conjunto de dados?
  
  > **R:** Predizer se um indivíduo sobreviveu, com base nos dados do embarque.

* **P:** Quais são os tipos de atributos existentes e qual é o atributo alvo?
    > **R:** Survived é o alvo e o restante ajudará a predizer.

    | nome_atributo | tipo_atributo | descricao |
    | :------------ |:--------------:|---------:|
    | PassengerId  | Qualitativo nominal| Apenas um ID |
    | Survived     | Qualitativo nominal| Sobreveu sim ou não |
    | Pclass       | Qualitativo ordinal| Classe econômica (1, 2 3) |
    | Name       | Qualitativo nominal| Nome do passageiro |
    | Sex       | Qualitativo nominal| Sexo do passageiro |
    | Age       | Quantitativo discreto| Idade do passageiro |
    | SibSp       | Quantitativo discreto| # de irmãos/conjuges a bordo |
    | Parch       | Quantitativo discreto| # de pais/filhos a bordo |
    | Ticket       | Qualitativo nominal| Número do ticket |
    | Fare       | Quantitativo contínuo| Taxa do passageiro |
    | Cabin       | Qualitativo nominal| Número da cabine |
    | Embarked       | Qualitativo nominal| Local de embarque |

<br>

* **P:** Quais são os problemas existentes? 
    > **R:** Não achamos uma correlação muito clara entre dois atributos, pela tabela de correlações, o que nos ajudaria na predição. Além disso, alguns campos no formato de
texto nos atrapalharam na hora de realizar predições.

<br>

>#### 2.2 Visão geral da base de dados em estudo:<br>

* **P:** seus dados são sobre o que?
    > **R:** Transtorno comum (Depressão)

* **P:** o que você deseja com este conjunto de dados? 
    > **R:** Predizer a presença do transtorno de depressão com base em perguntas respondidas por pacientes com comportamento considerado suficiente para diagnóstico do transtorno comum de depressão.
* **P:** quais são os tipos de atributos existentes e qual é o atributo alvo? 
    > **R:** O atributo alvo é o **dsm_mddh**, ele indica o diagnóstico realizado pelo entrevistador em relação a presença da depressão ou não. Considerando que são mais de 300 atributos, nós fizemos alguns pré-processamentos e selecionamos nesse primeiro momento os 10 melhores atributos. <br>Confira na tabela abaixo: <br>

    | nome_atributo | tipo_atributo | descricao |
    | :------------ |:-----------:| --------:|
    | SC20  | Qualitativo nominal| Had attack of fear/panic |
    | SC21  | Qualitativo nominal| Have you ever in your life had a period of tim... |
    | SC22  | Qualitativo nominal| Several days or longer felt discouraged about thing... |
    | SC23 | Qualitativo nominal| Several days or longer lost interest in things enjoyed |
    | SC25  | Qualitativo nominal| Several days or longer very irritable/grumpy/bad mood |
    | SC26  | Qualitativo nominal| Worried a lot more about things than other people |
    | CC4  | Quantitativo discreto| During the past 30 days, how many days did you... |
    | CC20A  | Qualitativo nominal| Problems getting to sleep past 12 months |
    | CC49B  | Quantitativo discreto| # visits to a psychiatrist past 12 months |
    | CC49C  | Quantitativo discreto| # visits to medical specialist like cardiologist pa... |


* **P:** quais são os problemas existentes? 
    > **R:** O maior problema foi a grande quantidade de NaN, principalmente nas questões que são diretamente relacionadas com a depressão, porém decidimos por não tratá-los diretamente e sim tentar convergir um modelo com as demais características que possuem poucou ou nenhum NaN. <br> 

* qualidade e clareza: garantir que a semântica dos atributos seja clara (nomes coerentes com os dados, se necessário renomear atributos).


### 3.Pré-processamento dos Datasets <br>

Realize o Pré-processamento e Tratamento de Dados em sua base/dataset.

>#### 3.1 Pré-processamento e tratamento na base de dados clássica:<br>
<ul>
    <li>nós eliminamos o campo <i>nome</i> da nossa base de treinos e testes, pois o nome completo
    iria nos atrapalhar na hora de realizar predições.</li>
    <li>Como o campo <i>cabin</i> possuía 687 valores nulos, nós também o desconsideramos.</li>
    <li>O campo <i>embarked</i> só possuía três valores distintos, então usamos um label encoder "manualmente" e transformamos
    os valores em números (1, 2 ou 3). Além disso, removemos os dois registros que possuíam um valor nulo para <i>embarked</i>.
    E o campo <i>embarked</i> foi retirado do dataset</li>
    <li>O campo <i>ticket</i> foi retirado do dataset por representar apenas um código de identificação da passagem.
    Sendo assim, este campo tem uma variação muito grande e acaba não ajudando o algoritmo na sua convergência.</li>
    <li>O campo <i>passengerId</i> foi retirado do dataset por representar apenas uma numeração dos passageiros
    (partindo do um até o numero máximo de passageiros). Sendo assim, este campo tem uma variação muito grande e acaba 
    não ajudando o algoritmo na sua convergência.</li>
    <li>O campo <i>sex</i> foi aplicado a técnica de <i>One Hot Encoding</i>, criando assim dois campos chamados <i>sex_female</i>
    e <i>sex_male</i>. Foi utilizado esta técnica para melhor predição do algoritmo. Além disso, o campo <i>sex</i> foi retirado do dataset.</li>
    <li>Foi imputado no campo <i>age</i> a média das idades deste mesmo campo, onde havia valores nulos.</li>
    <li>Foi aplicada a técnica de <i>Binning</i> no dataset na intenção da obtenção de melhores predições.</li>
    <li>Foi aplicada a técnica de <i>Balanceamento</i> no dataset na intenção da obtenção de melhores predições.</li>
    
</ul>

![cabin-nulos](https://user-images.githubusercontent.com/37307708/111407251-54cdb480-86b2-11eb-904e-9b6bfc4285e5.png)


<br>

>#### 3.2 Pré-processamento e tratamento na base de dados em estudo:<br>
##### Remoção de valores nulos
* Nós removemos as colunas completamente na presença de apenas um NaN ou mais. Não é a melhor abordagem mas foi feito para atender o prazo, esse é um corte bem significativo que deve ser avaliado antes da aplicação. Essa remoção de NaNs resultou numa redução de 337 atributos para 79.<br>
Note pela cor amarela a quantidade de valores nulos antes e depois.
###### Antes
![emestudo_heatmap_nan_antes](./images/emestudo_heatmap_nan1.png)

###### Depois
![emestudo_heatmap_nan_depois](./images/emestudo_heatmap_nan2.png)


##### Avaliando a presença de Outliers
* Nesta etapa "pegamos um atalho", ao invés de tratar os outliers de todas características disponíveis, nós usamos o método de seleção de características citado na próxima seção **Seleção de características**, ANTES para diminuir o volume de características que deveríamos tratar. 
* Consideramos o cenário em que os dados foram obtidos, nós ignoramos os atributos de valores nominais pois entendemos que seria muito difícil determinar um outlier. Dos atributos numéricos e contínuos nós observamos a presença de valores nominais misturados, nesse caso nós removemos todos os registros com valores nominais em atributos numéricos e contínuos. Essa operação resultou na redução de registros de 5037 para 4223.

Vale ressaltar que esse tipo de manobra é imprudente como um tiro no escuro. Por exemplo, imagine se tivéssemos um atributo com um potencial incrível para o modelo mas que ele está com alta quantidade de outliers, mas outros atributos que são menos interessantes para o modelo tem pouco outlier, é possível que ao aplicar uma seleção de características esse atributo de "potencial incrível" fique de fora. O inverso desse cenário também é verdadeiro dependendo do modelo!

>**SPOILER:** Alguns dos atributos que inicialmente se mostraram melhores, depois da remoção dos outliers, não foram mais relacionados entre as melhores características.

##### Balanceamento do Dataset
* O dataset estava bastante desbalanceado, do total de 4223 registros, 3455 representavam o diagnóstico negativo e apenas 768 para positivo. Sabendo disso, balanceamos o dataset em 50/50, sendo 768 registros de cada diagnóstico. Totalizando 1536 registros.

![emestudo_balance](./images/emestudo_balance.png)


##### Seleção de características
* Com o objetivo de diminuir ainda mais a quantidade de atributos que serão avaliadas pelo modelo, utilizamos o método de seleção de características do `sklearn` chamado `SelectKBest` combinado com `chi2` para selecionar os 10 melhores atributos dos 79. É um chute inicial ainda não sabemos se essa é uma quantidade suficiente para atingir boas métricas no modelo. Os atributos selecionados estão exibios na seção 2.1 do seu respectivo dataset.

### 4.Análise Exploratória dos datasets<br>
Explore conjunto de dados por meio de uma ferramenta (EDA), destacando em suas observações o que for considerado mais relevante.

>#### 4.1 Análise exploratória na base de dados clássica:<br>
Usando o Pandas Proffile, conseguimos obter algumas informações relevantes da nossa base clássica: 
[Report_Titanic.pdf](https://github.com/caicke/lap1/files/6153552/Report_Titanic.pdf)
![age](https://user-images.githubusercontent.com/37307708/111411718-d2e18980-86b9-11eb-88f7-ef8fc1db374e.png)
![cabin](https://user-images.githubusercontent.com/37307708/111411757-dffe7880-86b9-11eb-9a59-b6ba7e1df05f.png)
Como já havíamos observado, haviam muitos registros faltando nos campos <i>age</i> e <i>cabin</i>.<br>
Além disso, através do relatório vimos que o número de <i>zeros</i> nos campos <i>parch</i> (pais e filhos a bordo) e <i>sibsp</i> (irmãos/conjuges a bordo) eram maioria em seus respectivos campos (76,1% e 68,2%, respectivamente), provando que a maioria dos passageiros não possuíam nenhum tipo de parentesco entre eles.
![parch](https://user-images.githubusercontent.com/37307708/111412162-a5491000-86ba-11eb-8004-a80f161c95ee.png)
![sibsp](https://user-images.githubusercontent.com/37307708/111412511-4b951580-86bb-11eb-8568-68e50ffaf192.png)

>#### 4.2 Análise exploratória na base de dados em estudo:<br>

Sugestão: Utilizar ferramentas como Pandas Proffile e Sweetviz , Seaborn e Matplotlib <br>
    
[Tutorial básico com Seaborn](https://github.com/profmoisesomena/escience_and_tools/blob/master/seaborn/Seaborn_introduction.ipynb "Seaborn Introduction")
<br>

######Antes
Nós utilizamos o Pandas Profiling, porém o relatório dele é bastante extenso e não encaixa bem para essa explicação. Sendo assim nos contentaremos em explicar através do overview.

![emestudo_pprofile_overview](./images/emestudo_pprofile_overview.png)



* O **número de atributos** presente nesse dataset é bem grande, muito mais que o necessário para convergir um modelo. Por um lado é bom ter uma variedade de opções para escolher os melhores atributos, por outro isso aumenta bastante o trabalho de análise e pré-processamento;
* O **número de registros**, apesar do bom volume, apenas 1536 (30,49%) registros  foram utilizados porque estavam desbalanceados;
* A quantidade de **células faltando** é bem expressiva principalmente nas características mais diretas no assunto depressão;
* Dos **tipos de variáveis** a grande maioria é nominal, inclusive alguns atributos possuem valores categóricos e numéricos, nesses casos nós tratamos.

###### Depois

Considerando a análise anterior, o dataset foi tratado e um novo relatório foi gerado.

![emestudo_pprofile_overview](./images/emestudo_pprofile_overview_depois.png)

* A única consideração que gostaríamos de enfatizar neste relatório é a indicação de **registros duplicados** que acreditamos ser um erro.

![emestudo_pprofile_overview](./images/emestudo_pprofile_warnings_depois.png)

* Dos 10 atributos selecionados para o modelo, 2 apareceram no relatório de **Warnings** porém concordamos em desconsiderar o "aviso" pois é possível que os valores sejam 0 com bastante frequência nesses casos (``CC49B`` e ``CC49C``);
* Ainda existe uma preocupação em especial com ``SC2`` e ``CC5`` pois aparentam estar enviesados e de certa forma não tratamos esses casos. Podem ser atributos úteis para um modelo com mais de 10 atributos. 

<br>

># Marco de Entrega 01: Itens do Sprint 01 <br>
    
### 5.Estudo dos algoritmos previamente definidos para a pesquisa
  (explicação/teoria)<br>
  >#### 5.1 Visão geral sobre cada um dos algoritmos:<br>
    A) Explicação sobre o algoritmo/método de classificação adotado
    (como funciona, performance/complexidade para treino e para execução, etc...)
    B) Estudar e apresentar exemplo de aplicações com algoritmos
    C) Existem requisitos/premissas necessárias para aplicação do algoritmo, quais são?
    D) Aplicar os modelos estudados em bases de dados clássicas como Iris/Titanic 
    (no caso de desejar utilizar outra base consultar o professor)
    
>#### 5.2 Qual dos algoritmos estudados (não visão do grupo, com base nos resultados obtidos) é o mais recomendado para a base de dados clássica utilizada (explicar):<br>
>...
>#### 5.3 Qual dos algoritmos estudados (não visão do grupo) provavelmente será o mais recomendado para a base de dados em estudo (explicar):<br>
>...


># Marco de Entrega 02: Itens do Sprint 02 <br>
>

### 6.Implementar método no dataset em estudo  (explicação + datasets)<br>
    A) Explicação sobre o processo de aplicação dos algotítmos em estudo 
    no conjunto de dados em estudo (passos necessários/realizados)
    B) Implementar método nos datasets utilizados comparar resultados obtidos 
    e validar ou descartar hipótese do ítem 5.1 e 5.2.
    
>#### 6.1 Detalhamento dos processos de classificação com base nos algoritmos na base de dados em estudo:<br>
>...
>

### 7.Análise dos resultados obtidos <br>
    A) Detalhar conclusões com base nos resultados obtidos
    B) Definir quais trabalhos futuros podem ser realizados a partir das conclusões obtidas e tarefas realizadas.
    
>#### 7.1 Conclusões com base nos resultados obtidos:<br>
>...
>#### 7.2 Trabalhos futuros:<br>
>...
>
### 8. Resultados e Artefatos
>#### 8.1 Slides Finais
>#### 8.3 Demais artefatos solicitados pelo professor


># Marco de Entrega 03: Conclusão das atividades <br>

### 9 FORMATACAO NO GIT:<br> 
https://help.github.com/articles/basic-writing-and-formatting-syntax/
<comentario no git>
    
##### About Formatting
    https://help.github.com/articles/about-writing-and-formatting-on-github/
    
##### Basic Formatting in Git
    
    https://help.github.com/articles/basic-writing-and-formatting-syntax/#referencing-issues-and-pull-requests
    
    
##### Working with advanced formatting
    https://help.github.com/articles/working-with-advanced-formatting/
#### Mastering Markdown
    https://guides.github.com/features/mastering-markdown/

    
### OBSERVAÇÕES IMPORTANTES

#### Todos os arquivos que fazem parte do projeto (Imagens, pdfs, arquivos fonte, etc..), devem estar presentes no GIT. Os arquivos do projeto vigente não devem ser armazenados em quaisquer outras plataformas.
1. <strong>Caso existam arquivos com conteúdos sigilosos<strong>, comunicar o professor que definirá em conjunto com o grupo a melhor forma de armazenamento do arquivo.

#### Todos os grupos deverão fazer Fork deste repositório e dar permissões administrativas ao usuário do git "profmoisesomena", para acompanhamento do trabalho.

#### Os usuários criados no GIT devem possuir o nome de identificação do aluno (não serão aceitos nomes como Eu123, meuprojeto, pro456, etc). Em caso de dúvida comunicar o professor.

Link para curso de GIT<br>
![https://www.youtube.com/curso_git](https://www.youtube.com/playlist?list=PLo7sFyCeiGUdIyEmHdfbuD2eR4XPDqnN2?raw=true "Title")




