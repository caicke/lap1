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
##### Visão geral das bases de dados **Clássica** <br>
* Seus dados são sobre o que?
<br>**R**: Base de dados com informações sobre os tripulantes do Titanic.
* O que você deseja com este conjunto de dados?
<br>**R**: Predizer se um indivíduo sobreviveu, com base nos dados do embarque.
* Quais são os tipos de atributos existentes e qual é o atributo alvo?
<br>**R**: Survived é o alvo e o restante ajudará a predizer.

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

* Quais são os problemas existentes? <br>
**R**: Não achamos uma correlação muito clara entre dois atributos, pela tabela de correlações, o que nos ajudaria na predição. Além disso, alguns campos no formato de
texto nos atrapalharam na hora de realizar predições.

<br>
##### Visão geral das bases de dados **Em estudo** <br>

* seus dados são sobre o que? <br> **R**: Transtorno comum (Depressão)
* o que você deseja com este conjunto de dados? <br> **R:** Classificar a presença do transtorno de depressão com base em perguntas respondidas por pacientes que possuem potencial comportamento para diagnóstico da depressão.
* quais são os tipos de atributos existentes e qual é o atributo alvo? <br> **R:** TBD<br>

| representação_do_atributo | tipo_atributo | descrição |
| :------------ |:-----------:| --------:|
| 1  | Qualitativo nominal| SIM |
| 5  | Qualitativo nominal| NAO |
| 8 ou 998 | Qualitativo nominal| NAO SABE |
| 9 ou 999  | Qualitativo nominal| RECUSOU |


* quais são os problemas existentes? **R:** Muitos NaN, dominínio de valores dos atributos conflitantes<br> 
* qualidade e clareza: garantir que a semântica dos atributos seja clara (nomes coerentes com os dados, se necessário renomear atributos).

>#### 2.1 Visão geral da base de dados clássica:<br>


>#### 2.2 Visão geral da base de dados em estudo:<br>



### 3.Pré-processamento dos Datasets <br>

Realize o Pré-processamento e Tratamento de Dados em sua base/dataset.

#### 3.1 Pré-processamento e tratamento na base de dados clássica:<br>
<ul>
    <li>nós eliminamos o campo <i>nome</i> da nossa base de treinos e testes, pois o nome completo
    iria nos atrapalhar na hora de realizar predições.</li>
    <li>Como o campo <i>cabin</i> possuía 687 valores nulos, nós também o desconsideramos.</li>
    <li>O campo <i>embarked</i> só possuía três valores distintos, então usamos um label encoder "manualmente" e transformamos
    os valores em números (1, 2 ou 3). Além disso, removemos os dois registros que possuíam um valor nulo para <i>embarked</i>.</li>
    
</ul>
![cabin-nulos](https://user-images.githubusercontent.com/37307708/111407251-54cdb480-86b2-11eb-904e-9b6bfc4285e5.png)


>#### 3.2 Pré-processamento e tratamento na base de dados em estudo:<br>
>...    

### 4.Análise Exploratória dos datasets<br>
Explore conjunto de dados por meio de uma ferramenta (EDA), destacando em suas observações o que for considerado mais relevante.

#### 4.1 Análise exploratória na base de dados clássica:<br>
Usando o Pandas Proffile, conseguimos obter algumas informações relevantes da nossa base clássica: 
[Report_Titanic.pdf](https://github.com/caicke/lap1/files/6153552/Report_Titanic.pdf)
![age](https://user-images.githubusercontent.com/37307708/111411718-d2e18980-86b9-11eb-88f7-ef8fc1db374e.png)
![cabin](https://user-images.githubusercontent.com/37307708/111411757-dffe7880-86b9-11eb-9a59-b6ba7e1df05f.png)
Como já havíamos observado, haviam muitos registros faltando nos campos <i>age</i> e <i>cabin</i>.<br>
Além disso, através do relatório vimos que o número de <i>zeros</i> nos campos <i>parch</i> (pais e filhos a bordo) e <i>sibsp</i> (irmãos/conjuges a bordo) eram maioria em seus respectivos campos (76,1% e 68,2%, respectivamente), provando que a maioria dos passageiros não possuíam nenhum tipo de parentesco entre eles.
![parch](https://user-images.githubusercontent.com/37307708/111412162-a5491000-86ba-11eb-8004-a80f161c95ee.png)
![sibsp](https://user-images.githubusercontent.com/37307708/111412511-4b951580-86bb-11eb-8568-68e50ffaf192.png)

>#### 4.2 Análise exploratória na base de dados em estudo:<br>
>...    
Sugestão: Utilizar ferramentas como Pandas Proffile e Sweetviz , Seaborn e Matplotlib <br>
    
[Tutorial básico com Seaborn](https://github.com/profmoisesomena/escience_and_tools/blob/master/seaborn/Seaborn_introduction.ipynb "Seaborn Introduction")

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




