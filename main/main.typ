#import "../main/template.typ": project

#show link: underline

// #import "../main/appendices.typ" : appendices
// #show: appendices.with()

#show: project.with(
  title: "--",
  abstract: [
    --
    ],
  keywords: ("--", "--", "--", "--"),
  team-number: "00000",
  problem-chosen: "A",
  year: "2024",
  bibliography-file: "refs.bib",
)


= Introduction

== Background

During the 2024 Summer Olympics in Paris, spectators and the media will focus on individual event performances and pay close attention to each country's overall medal table ranking. The medal table reflects not only the efforts of individual athletes and teams but also the overall strength and competitiveness of countries in the field of sports.

Before the start of the Olympic Games, many organizations and experts try to predict the outcome of the medal table. These predictions are usually based on historical data, recent event performances, athlete rosters, and the advantages of the host country. However, accurate medal predictions are not an easy task as they require a combination of complex factors such as athlete status, unforeseen circumstances during the competition, and changes in the program settings.

Medal predictions can help us to provide a basis for national sports planning, helping to rationally allocate resources, optimize project development, and enhance overall sports strength. At the same time, it can motivate athletes and coaches to set clear goals, adjust training strategies, and enhance confidence. It can promote the development of the sports industry. It also provides a reference for sports research and analysis, revealing the trend of changes in sports strength and potential influencing factors of various countries.


// == Literature Review


== Restatement of the Problem

// - #[
- Task 1: We need to develop a model of the total number of medals for each country. Based on this model, we need to predict the prediction intervals for all outcomes of the 2028 Summer Olympics medal table in Los Angeles, USA.   
// ]

- Task 2: According to the model, we need to analyse which sports are most important to each country and the impact of the sports chosen by the host country on the outcome of the competition.
- Task 3: For countries that have not won medals, we need to predict how many countries will win their first medals at the next Olympics, giving the chances of this estimate being accurate.
- Task 4: We need to study the data for evidence of changes that may be caused by the “great coach” effect. Estimate the contribution of this effect to the number of medals. Select three countries and identify the sports in which they should consider investing in “great” coaches and estimate the effect.
- Task 5: Analyse what other factors may affect Olympic medal counts based on the modeling model.
// Develop a model of the total number of medals for each country.

// - Based on your model, predict the prediction intervals for all outcomes of the 2028 Summer Olympics medal table in Los Angeles, USA. Analyse which countries are most likely to improve and which countries will perform worse.
// - Predict how many countries will win their first medals at the next Olympics, giving the chances of this estimate being accurate.
// - Analyse which sports are most important to each country based on the model. Analyze the impact of the sports chosen by the host country on the outcome of the competition.
// - Study the data for evidence of changes that may be caused by the“great coach” effect. Estimate the contribution of this effect to the number of medals. Select three countries and identify the sports in which they should consider investing in “great” coaches and estimate the effect.
// - Analyse what other factors may affect Olympic medal counts based on the modeling model

#pagebreak()

= Task 1: Model for Medal Prediction

//在本任务中，我们建立了一个基于梯度提升的集成模型。

In this task, we developed an  ensemble model based on gradient boosting to predict the total number of medals for each country. 

// 经过多次尝试和参数调整，我们模型在训练集和测试集上的$R^2$分别达到了$1.00$和$0.96$，并且将数据集上的MSE控制在了$2.77$左右，表现较为优秀。

After multiple attempts for parameter adjustment and train , our model achieved an $R^2$ of $1.00$ and $0.96$ on the training and test sets, respectively, and controlled the `MSE` on the dataset to around $2.77$, `MAPE` to around $12.93%$ , showing relatively good performance.

// 并以此预测了2028年洛杉矶奥运会的奖牌榜：美国将以45枚金牌、46枚银牌和25枚铜牌，共计117枚奖牌的成绩保持第一，而中国将以35枚金牌、27枚银牌和21枚铜牌，共计83枚奖牌的成绩位居第二。

According to that, we predicted the medal table for the 2028 Los Angeles Olympics: the United States will remain in first place with 45 gold medals, 46 silver medals, and 25 bronze medals, totaling 117 medals; while China will rank second with 35 gold medals, 27 silver medals, and 21 bronze medals, totaling 83 medals.

== Assumptions and Justification
// 假设与依据


// 考虑到奥运会的奖牌数具有复杂性，充分考虑其所有因素显然是不现实的。因此，我们在研究了给定的数据集之后，做出了以下假设：

Considered about the complexity of the Olympic medal count, it is obviously unrealistic to consider all factors. Therefore, after studying the given dataset, we made the following assumptions:

// 决定性： 各国的奖牌数主要取决于其运动员的实力，同时，该国当年的奖牌数可以一定程度上反映该国当年的实力。
*Deterministic*: The number of medals in each country mainly depends on the strength of its athletes, and the number of medals in that country in that year can reflect the strength of that country to a certain extent.

// 稳定性： 虽然奖牌数由多种外部因素（如国际政治）的影响，但是我们认为在短时间内，这些因素的影响是稳定的。
*Stability*: Although the number of medals is affected by various external factors (such as international politics), we believe that the impact of these factors is stable in the short term.

// 主场优势： 主办国在奥运会上通常会取得更好的成绩。
*Host Advantage*: The host country usually performs better at the Olympics.

// == Notations
// // 符号




== Model Overview
// 模型概述

// 在尝试了多种模型管道结构与特征工程方法后，我们最终选择了以梯度提升为核心的一种集成算法实现奖牌数预测。

After trying lots of model pipelines and feature engineering methods, we finally chose an ensemble algorithm with *gradient boosting* as the core to predict the number of medals.

// 我们的训练管道由Figure 1所描述。
As Figure 1 describes, our training pipeline consists of the following steps:

// #figure(
//   image("pipeline.png", width=80%),
// )

#figure(
  image("pipeline.png",width: 80%),
  caption: [Modeling Pipeline],
)
// Figure 1: Model Pipeline

// 则可以将模型表示为：
The model can be represented as:

$
  hat(y) = sum_(t=1) ^T f_t (X) , f_t in cal(F)
$

// 其中，$cal(F)$是基学习器的树结构空间。
Where $cal(F)$ is the tree structure space of the base learner.

// 在具体叙述模型的数学表示之前，我们先规定参数：$gamma$为叶子节点分裂的最小增益，$lambda$为叶子节点的L2正则化系数。
Before specifically describing the mathematical representation of the model, we first define the parameters:
- $gamma$ is the minimum gain of the leaf node split
- $lambda$ is the L2 regularization coefficient of the leaf node.
- $LL$ is the loss function, which is defined as:

$
LL(x,y) = (x-y)^2/2
$

// 那么，基于传统的GBDT算法，我们引入二阶泰勒展开近似损失函数：@2
Then, based on the *traditional GBDT algorithm* , we introduce the second-order Taylor expansion to approximate the loss function:


$
frak(J)_("Obj") \ ^((t)) approx sum_(i=1)^n [g_i f_t(x_i) + 1/2 h_i f_t^2 (x_i)] + Omega (f_t)  \
Omega(f) = gamma T +1/2 lambda ||w_j|| ^2 ,\
g_i = partial_(hat(y)^(t-1)) LL(y_i, hat(y)^(t-1)), \
h_i = partial_(hat(y)^(t-1)) ^2 LL(y_i, hat(y)^(t-1)), \
$

// 在梯度提升的树生成过程中，我们采用贪心分裂策略，通过最小化损失函数的增益来选择最佳的分裂点。
In the process of tree generation in gradient boosting, we adopt a greedy split strategy to select the best split point by minimizing the gain of the loss function.

$
g(I) = 1/2 [g_I^2/(h_I + lambda) + g_L^2/(h_L + lambda) + g_R^2/(h_R + lambda) - g^2/(h + lambda)] - gamma \
GG = sum_(I=1)^T [g(I)^2/(h_I + lambda)] + gamma T
$

// 其中，$I$表示当前节点，$L$和$R$分别表示分裂后的左右子节点。
Where $I$ represents the current node, and $L$ and $R$ represent the left and right child nodes after the split.

// 通过按$hat(y)_i^((t)) = hat(y)_i^((t-1)) + eta f_t(x_i)$ 重复地将新树加入模型，我们最终得到了一个强大的集成模型。
By repeatedly adding new trees to the model according to $hat(y)_i^((t)) = hat(y)_i^((t-1)) + eta f_t(x_i)$, we finally obtained a powerful ensemble model.

== Faucets Determination
// 参数确定

// 在模型假设中，我们提到，一个国家的奖牌数与其“实力”强相关。那么，我们需要得到一个（或者一些）能够反映国家实力的指标，作为我们模型的特征。
As we mentioned in the model assumptions, the number of medals in a country is strongly related to its "strength". Therefore, we need to obtain an index (or some) that can reflect the strength of a country as a feature of our model.

// 分析奖牌榜上靠前的国家，我们发现这些国家都有一些“统治力”较强的项目，如美国的游泳、田径，中国的举重、乒乓球，俄罗斯的体操、跳水等。这些项目为其的奖牌总榜贡献远超其他项目。
After analyzed the countries at the top of the medal table, we found that these countries all have some projects with strong "dominance", such as swimming and track and field in the United States, weightlifting and table tennis in China, gymnastics and diving in Russia, etc. These projects contribute far more to their total medal list than other projects.

// 为了将这一因素纳入模型，我们引入以下的一些特征：
To incorporate this factor into the model, we introduce the following features:

// - 该国家统治力较高的一些项目，其“统治力”的数值。
- The *dominance* value of some projects with high dominance in that country.
// - 该国家的"偏科程度"
- The *degree of Specialization* of that country.
// - 该国家历史上统治较高的一些项目，在本届奥运会能提供的奖牌数
- The number of medals that the country can provide in this session for some projects with high dominance in its history.

// 其中：统治力 $cal(D)$ 定义为：

The *dominance* $cal(D)$ is defined as:

$
cal(D) = "gained"_i/"total"_i , forall i in S
$

// $S$为该国在该届参加的所有项目的集合，$"gained"_i$为该国在该项目上获得的奖牌数，$"total"_i$为该项目的总奖牌数。
Where $S$ is the set of all projects that the country participated in this session, $"gained"_i$ is the number of medals won by the country in that project, and $"total"_i$ is the total number of medals in that project.

// 偏科程度 $cal(V)$ 定义为所有项目统治力的方差：
*Degree of Specialization* $cal(V)$ is defined as the variance of the dominance of all projects:

$
cal(V) = "var"(cal(S))
$

// 同时，我们加入了该年奥运会的主办方作为参数，以应对主场优势对奖牌数的影响。
At the same time, we added the host of the Olympic Games as a parameter to deal with the impact of the host advantage on the number of medals.

== Data Cleaning and Preprocessing

// 我们按照以下顺序对数据进行了清洗和预处理。
We cleaned and preprocessed the data in the following order:

// - 合并国家代码，并建立国家代码与国家名称的映射。
- Merge the country codes and establish a mapping between the country codes and country names.
// - 去除由于冬奥会、战争等原因造成的缺失值。
- Remove missing values caused by the Winter Olympics, wars, etc.
// - 添加了$2028$年的项目数据。
- Added project data for $2028$.//@1
// - 通过建立好的映射表，根据详细的运动员数据计算了各国家各项目的详细奖牌数据。
- According to the detailed athlete data, we calculated the detailed medal data of each country and each project through the established mapping table.
// - 根据以上映射表，计算了各国家的统治力、偏科程度。
- According to the above mapping table, we calculated the dominance and degree of specialization of each country.

== Model Training

// 我们使用了`tpot`库中的`XGBRegressor`和网格化调参来自动化训练过程，并对其进行了高度封装。
We used the `XGBRegressor` in the `tpot` library and grid search to automate the training process and highly encapsulated it.

// 基于这些封装，我们能够自动化地调整一些变量，以找到最优的模型。
Based on these encapsulations, we can automatically adjust some variables to find the optimal model.

//规定两个描述特征组合的参数：$NN$和$MM$。
Two parameters describing the feature combination are defined: $NN$ and $MM$, where:

// $NN$代表参数中含有的往届数据的年数。
- $NN$ represents the number of years of historical data in the parameters.

// $MM$代表参数中取“统治力”较高的项目的个数。
- $MM$ represents the number of projects with high "dominance" in the parameters.

// 以此绘制了MAPE随$NN$和$MM$变化的热度图：
We then plotted a heatmap of *MAPE*( Mean Absolute Percentage Error) against $NN$ and $MM$:

// Figure 2: MAPE Heatmap
#figure(
  image("MAPE Heatmap.png",width: 70%),
  caption: [MAPE Heatmap],
)

// 最终，我们选择了$NN=3$，$MM=2$作为最终的特征组合。
According to results above, we chose $NN=3$ and $MM=2$ as the final feature combination.

== Model Evaluation

// 经过一些时间的训练，我们的模型在总奖牌数量的预测上，达到了令人激动的效果。
After some time of more in-depth training, our model achieved exciting results in predicting the total number of medals.

#figure(
  image("Total.png",width: 70%),
  caption: [Prediction vs. True Value for Total Medals],
)

// Figure 3: Prediction vs. True Value

// 更多地，我们计算了模型的其他数据：
Furthermore, we calculated other evaluation data of the model:

#figure(
  table(
  columns: (3fr,3fr,3fr,3fr,3fr),
  inset: 3pt,
  stroke:none,
  align: (left,left,left,left,left),
    [*SSE*],[*MSE*],[*MAE*],[*R^2*],[*MAPE*],
    [0.002],[2.77],[0.23],[0.96],[12.93%],
  ),
  caption: [Model Performance],
)<tab1>\

// Table 1: Model Performance

// 并且评估了各个参数的排列重要性：
And evaluated the importance of the arrangement of various parameters:
#figure(
  image("Prediction vs. True Value.png",width: 70%),
  caption: [Permutation Importance for each Feature],
)

// 图表显示，本国统治力最高的项目的奖牌数、本国的历史两年奖牌数在模型中起到了最为重要的作用。
The chart shows that the number of medals in the country's most dominant project and the number of medals in the country's history in the past two years play the most important role in the model.

// 因此我们认为，本国的历史两年奖牌数可以较好地反映本国的实力，而本国统治力最高的项目的奖牌数则会决定该国能否将其实力转化为奖牌。
Therefore, we believe that the number of medals in the country's history in the past two years can better reflect the strength of the country, while the number of medals in the country's most dominant project will determine whether the country can convert its strength into medals.

// Figure 4: Feature Importance

// 最后，我们计算出了以$R^2$为评分和以百分比误差为评分的$95%$置信区间分别为$[0.844, 1]$和$[0,1.021%]$，这意味着我们的模型在整体预测中稳定性较高，并且有较高的把握确保预测结果的准确性。
Finally, we calculated that the $95%$ confidence intervals for $R^2$ and percentage error as the score are $[0.844, 1]$ and $[0,1.021%]$, respectively. This means that our model has high stability in overall prediction and has a high degree of certainty to ensure the accuracy of the prediction results.

// 自动化上述过程，我们分别得到了铜牌、银牌、金牌的预测结果。
Automating the above process, we obtained the prediction results for bronze, silver, and gold medals, respectively.

#figure(
  image("模型合集.png",width: 100%),
  caption: [Intuitive Performance of each Model],
)

// 以上所有评估过程都将预测数据进行了取整。
*All the above evaluation processes have rounded the predicted data.*

== Results
// 结果

// 通过代入$2028$年的数据，我们得到了以下的预测结果的原始数据：
By substituting the data for $2028$, we obtained the original data of the following prediction results:


#figure(
  table(
  columns: (2fr,3fr,3fr,3fr,3fr,2fr,3fr),
  inset: 3pt,
  stroke:none,
  align: (center,left,center),
[*rank*],[*Gold*],[*Silver*],[*Bronze*],[*Total*],[*NOC*],[*Year*],
[1],[45.990759],[46.052414],[25.373566],[117.41674],[USA],[2028],
[2],[35.17961],[27.126156],[21.585402],[83.89117],[CHN],[2028],
[3],[21.00542],[21.780415],[22.603785],[65.38962],[GBR],[2028],
[4],[25.49453],[21.445782],[14.94941],[61.88972],[AUS],[2028],
[5],[34.74799],[8.688086],[10.723672],[54.159744],[JPN],[2028],
[6],[17.675909],[11.654378],[13.032905],[42.363194],[FRA],[2028],
[7],[9.2394495],[9.359339],[9.370603],[27.96939],[KOR],[2028],
[8],[9.753317],[5.24933],[12.4193325],[27.42198],[ITA],[2028],
[9],[14.714196],[4.5507064],[7.990886],[27.255789],[ESP],[2028],
[10],[10.101428],[6.1777368],[10.032182],[26.311348],[NED],[2028],


  ),
  caption: [Medal Prediction for 2028 Summer Olympics],
)<tab1>\

// Table 2: Medal Prediction for 2028 Summer Olympics

#pagebreak()

= Task 2: Analysis of Important Sports

// 在这个任务中，我们建立了一个与Task.1类似的机器学习模型，并使用参数的重要性来分析项目对于各国奖牌数的影响。
In this task, we established a machine learning model similar to *Task.1* and used the importance of parameters to analyze the impact of projects on the number of medals in each country.

// == Model Overview

== Determination of Parameters

// 在这次，我们将所有项目的统治力均作为参数引入了模型中。
This time, we introduced the dominance of all projects as parameters into the model.

== Model Evaluation

// 与Task.1 类似，我们首先对这个模型的效果进行一个简要的评估：
Same as *Task.1*, we first made a brief evaluation of the effect of this model:
#figure(
  image("forSport_TP.png",width: 70%),
  caption: [Prediction vs. True Value for Sport],
)
// Figure 5: Prediction vs. True Value forSport_TP.png
#pagebreak()
#figure(
  table(
  columns: (3fr,3fr,3fr,3fr,3fr),
  inset: 3pt,
  stroke:none,
  align: (left,left,left,left,left),
    [*SSE*],[*MSE*],[*MAE*],[*R^2*],[*MAPE*],
    [0.31],[26.72],[1.79],[0.88],[61.52%],
  ),
  caption: [Model Performance],
)<tab1>\

// 虽然模型拟合结果与Task.1相比有所下降，但是从图表上看，模型的预测结果与真实值的拟合程度仍然较高。因此我们认为，模型能够有效地提取项目对于各国奖牌数的影响。
Although the fitting results of the model have decreased compared to *Task.1*, the degree of fitting between the predicted results of the model and the true values is still relatively high from the chart. Therefore, we believe that the model can effectively extract the impact of projects on the number of medals in each country.

// 那么，我们依然可以通过计算各个参数的排列重要性，来分析项目对于各国奖牌数的影响。
Therefore, we can still analyze the impact of projects on the number of medals in each country by calculating the importance of the arrangement of various parameters.

// Table 3: Model Performance

== Results

// 通过计算各个参数的排列重要性，并取其前$10$个，我们得到了以下的结果：
By calculating the importance of the arrangement of various parameters and taking the top $10$, we obtained the following results:

#figure(
  image("forSport.png",width: 70%),
  caption: [Permutation Importance for each Feature],
)

// Figure 6: Feature Importance forSport.png

// 可以看出，各国的奖牌数主要与其在游泳、田径、体操、举重、乒乓球等传统大项上的表现有关。这也与我们的预期相符。
It can be seen that the number of medals in each country is mainly related to its performance in traditional events such as swimming, track and field, gymnastics, weightlifting, and table tennis. This is also consistent with our expectations.

// 我们可以取一个例子。例如，如果美国将其在乒乓球上的统治力提高$10\%$，那么其奖牌数将会增加$2.35$枚。

// // 参数确定
#pagebreak()
= Task 3: Prediction of First-time Medal Winners

== Model Construction

// 尝试构建神经网络模型
Attempt to Build a Neural Network Model

// 构建神经网络模型以适应历史上首次获奖国家的特征
A neural network model is constructed to fit the characteristics of countries that have won their first medal in past years.

Let $X = [x_1, x_2, x_3]$. Here:
#figure(
  table(
    columns: (1.2fr, 4fr),
    inset: 4pt,
    stroke: none,
    align: (center, left),
    [*Symbols*], [*Description*],
    [$x_1$], [The number of editions in which the country has participated without winning any medals since its first participation.],
    [$x_2$], [The number of athletes the country has in this edition],
    [$x_3$], [The average number of medals awarded in this edition.]
  ),
  caption: [Variable Definitions],
)<tab-symbols>

// 使用X作为输入层，建立两个隐藏层（4/3神经元），输出层1神经元表示获奖概率
Use $X$ as the input layer, establish two hidden layers with 4 and 3 neurons respectively, and set the output layer with 1 neuron to represent the probability of winning a medal.

Define $y$ as the medal-winning indicator:
$y = cases(
  0 quad "indicates no medal",
  1 quad "indicates a medal is won"
)$

== Model Solution

=== Data Preparation
// 基于官网提供文件处理数据集，训练测试集划分比例0.2
The dataset is processed based on the files provided on the official website. The samples are split into training and testing sets with a ratio of 0.2.

#figure(
  image("C_3data initialization.svg", width: 90%),
  caption: [Data Preparation Flowchart],
)<fig-flow>

=== Neural Network Framework Construction
// 神经网络框架构建
Let $a^{(i)}$ represent the activation value of the i-th layer:

$
a^{(2)} = g(Theta^{(1)} X) \
a^{(3)} = g(Theta^{(2)} a^{(2)}) \
a^{(4)} = g(Theta^{(3)} a^{(3)}) \
hat(y) = a^{(4)}
$

Where:
- $Theta^{(i)}$: Propagation matrix
- $g$: Sigmoid activation function with output range $(0,1)$

=== Cost Function
// 二元交叉熵损失函数
The binary cross-entropy loss function measures the difference between predicted and actual values:

$
J(Theta) = -1/n sum_(i=1)^n [y^((i)) log(hat(y)^((i))) + (1 - hat(y)^((i))) log(1 - y^((i)))]
$

=== Problem Transformation
// 通过样本学习优化参数来最小化损失函数
The model's learning involves minimizing the loss function by optimizing parameters through sample-based learning.

=== Parameter Optimization
==== Optimization Methods
// 使用梯度下降法优化损失函数
Use gradient descent method with backpropagation:

Update rule:
$
Theta_(j  k)^(i) := Theta_(j k)^(i) - eta ("del" J)/("del" Theta_(j k)^(i))
$

// 采用分组平均下降策略
With grouped averaging approach:
- Let $m$ = number of sample groups
- Process each group's average gradient per iteration

==== Model Training
// 主要训练步骤
*Training Procedure*:
1. Initialize parameters $Theta^(1)$, $Theta^(2)$, $Theta^(3)$ and hyperparameters
2. Forward propagation: Compute $a^(2)$, $a^(3)$, $a^(4)$
3. Split samples into $m$ groups
4. Backpropagation: Calculate gradients per group
5. Update parameters using group averages
6. Record loss values
7. Repeat until convergence or max epochs

== Model Evaluation
// 观察学习曲线与准确率曲线
#figure(
  image("C_2_loss_Accuracy.png", width: 90%),
  caption: [Learning Curve and Accuracy],
)<fig-eval>

// 最终预测结果展示
#figure(
  image("The top 10 countries most probable to win first medal.png", width: 80%),
  caption: [Top 10 Potential First-time Medal Winners],
)<fig-results>

[*Note*]: Only displaying countries with probability > 0.1 for clarity.

#pagebreak()

= Task 4: Analysis of the "Great Coach" Effect

== Main tasks

We used mathematical modeling to determine the point when a coach came on board and quantify the contribution of coach turnover to the medal count based on medal data `summerOly_athletes.csv` for Olympic athletes through a change point detection method. We then recorded the country's level before the point of change in a given sport (the detailed quantification process is written in #link(<Quantifying>)[Quantifying]). By analyzing the historical level as well as the total number of medals in the historical sport, three countries were selected to find the sports in which they should consider investing in “Great Coach” and to estimate their impact.

// 我们通过数学建模，基于奥运会运动员的奖牌数据（summerOly_athletes.csv），通过变化点检测方法确定教练上任的时间点和量化教练更替对奖牌数的贡献。然后我们记录下该国家在某个项目的变化点前的水平（详细的量化过程写在了2.5.2），通过分析历史水平以及历史项目的总奖牌数，选择出三个国家，找到它们应考虑投资于 “优秀 ”教练的运动项目，并估计其影响。

== Data preprocessing

We take the medal count (gold, silver, bronze) of a country in a given sport at each Olympics. Because the types of medals (gold, silver, bronze) reflect different variations in the strength of a country in a particular sport, we assign different weights to each type of medal: Gold: 3; Silver: 1; Bronze: 0.5. then: 

$
  W_t=3*"Gold"_t+1*"Silver"_t+1*"Bronze"_t
$

Where:
- $W_t$ is the weighted total number of medals in a sport for that country in year $t$.
- $"Bronze"_t$ is the number of bronze medals won by the country in a sport in year $t$.
- $"Silver"_t$ is the number of silver medals won by the country in a sport in year $t$.
- $"Gold"_t$ is the number of gold medals won by the country in a sport in year $t$.

== Bayesian change point detection

=== Principle

*Bayesian change point detection* is a method based on Bayesian statistics for identifying change points in time series data. A change point is a point where there is a significant change in the distribution of the data, such as a change in the mean, variance, or trend. Bayesian methods can deal with the change point detection problem flexibly by introducing prior and posterior distributions and providing uncertainty estimates of the location of change points.//@5-1

=== Model Assumption

Assume that the weighted medal count $W_t$ in year $t$ is distributed before and after the change point obeys a normal distribution with different means and variances:

$
  W_t prop NN(mu_k, sigma_k^2), t in [t_(k-1), t_k]
$
/*
μ_k is the mean of the kth interval
σ_k^2 is the variance of the kth interval

*/
Where:
- $mu_k$ is the mean of the $k$th interval.
- $sigma_k^2$ is the variance of the $k$th interval.


=== Model Solution

By analyzing the data$W_1, W_2,..., W_t,$ estimating the location of the change point $t_1,t_2,...,t_k$ and the mean and variance corresponding to the change point to detect the change point. According to the Bayesian formula, the posterior probability is：

$
p(t_k | W) = p(W | t_k) p(t_k) / p(W)
$

Where:
- $p(t_k | W)$ is  likelihood function of the data at change point $t_k$.
- $p(t_k)$ is prior distribution of the change point $t_k$, assuming that the probability of the change occurring is uniform, i.e. $p(t_k) = 1/T$.
- $p(W)$ is marginal likelihood of the data.

By calculating the posterior probability for each year, we can detect the year of change in the data. Bayesian change point detection can be modeled using the changefinder library in Python. The detection function in changefinder returns a list *Change* containing the change point detection results for each year. Each element is a boolean value indicating whether the year is a change point: True means the year is a change point; False means the year is not a change point. Using the list *Change*, it is possible to identify which years are the years in which the number of medals changed significantly, and thus infer the impact of coaching changes or other factors on the number of medals. We denote the degree of change in the medal count for that year by:

$
  "Change"_t times "Standardized"(W_t)
$
/*Where the standardized weighted medal count is calculated:

Standardized_W_t=(W_t-mean)/std
mean is the mean of W_t; std is the variance of W_t
*/

The standardized weighted medal count is calculated as:

$
  "Standardized"(W_t) = (W_t - "mean") / "std"
$

Where:
- $"mean"$ is the mean of $W_t$.
- $"std"$ is the variance of $W_t$.

=== Analysis of Results

Using Python, we draw a graph of the change in the weighted number of medals for a given country, and the change in the degree of change. The point of change and the degree of change in the values were also derived. The results are given below:

// Figure1: BRA Football Weighted Medals & Degree of Change
#figure(
  image("Figure1 BRA Football Weighted Medals & Degree of Change.png", width: 80%),
  caption: [BRA Football Weighted Medals & Degree of Change],
)

One of the change points (2020) caught our attention, which had a degree of change of $15.66$. Tite became the coach of the Brazilian soccer team in 2016, and he led the team to successfully defend its title by winning the gold medal at the 2020 Olympic Games in Tokyo. Tite continues to keep Brazilian soccer competitive at the international level with his flexible tactical adjustments and precise grasp of the players' psychology. /*@5.2*/ As we analyze this information, we can argue that Tite's coaching is the reason for the Brazilian soccer team's surge in medals in 2020.

#figure(
  image("Figure2 USA Gymnastics Weighted Medals & Degree of Change.png", width: 80%),
  caption: [USA Gymnastics Weighted Medals & Degree of Change],
)

One of the change points (2012) caught our attention, which had a degree of change of 12.20. Béla Károlyi became the coach of the USA Gymnastics team in 1999, introducing Romanian training methods and improving the overall level of the USA Gymnastics team. /*@5.3*/ When we analyze this information, we can conclude that Béla Károlyi's coaching is the reason for the surge in the number of medals of the U.S. Gymnastics team in 2012.

#figure(
  image("Figure3 NED Cycling Weighted Medals & Degree of Change.png", width: 80%),
  caption: [NED Cycling Weighted Medals & Degree of Change],
)

One of the change points 2000 caught our attention, this point has a degree of change of 13.50. Max van der Stoep became the coach of the Dutch cycling team in 2000. Under Max van der Stoep's coaching, the Dutch cycling team performed well in the 2004 Olympic Games in Athens and the 2008 Olympic Games in Beijing, winning several medals. The Dutch team has made significant progress in track cycling, becoming one of the world's strongest teams in the sport. /*@5.3*/ When we analyze this information, we can conclude that Max van der Stoep's coaching was the reason for the Dutch cycling team's surge in medals in 2012.

The three data sets described above are evidence of changes that may be caused by the *“great coach”* effect. Next, we will quantify the contribution of this effect to medal counts.

== Quantifying the level of a country in a given project <Quantifying>

=== Calculating a Coach's Contribution

We use weighted medal counts to measure the difference between before and after a coach takes office. The average of the weighted medal counts was first calculated.

/*
Average of weighted medal counts before the change point: W_average_before=(1/T_before)*sum(W_t) t∈before
Average of weighted medal counts after the change point: W_average_after(1/T_after)*sum(W_t) t∈after
T_before is the number of years before the change point.
T_after is the number of years after the change point.

*/

The average of the weighted medal counts before the change point is calculated as:
$
 W_"average,before" = (1/T_"before") sum_(t in "before")(W_t)
$

The average of the weighted medal counts after the change point is calculated as:
$
  W_"average,after" = (1/T_"after") sum_(t in "after")(W_t)
$

Where:
- $T_"before"$ is the number of years before the change point.
- $T_"after"$ is the number of years after the change point.

We can calculate the contribution rate of a coach by comparing the change in the number of medals before and after the change point. Define the contribution rate as：

$
  "Contribution" = (W_"average,after" - W_"average,before") / W_"average,before"
$

If the contribution rate is greater than $0$, it means that the arrival of the coach has had a positive impact on the number of medals. The contribution rate reflects the magnitude of the increase in the number of medals, with larger values indicating a more significant change, i.e., the greater the change caused by the “great coach” effect.

=== Analysis of Results

The results can be calculated through Python and are as follows:

/*BRA Football Contribution: 1.25
USA Gymnastics Contribution: 0.51
NED Cycling Contribution: 1.71
*/

#figure(
  table(
  columns: (2fr,2fr,3fr),
  inset: 3pt,
  stroke:none,
  align: (right,center,left),
  [*Country*],[*Sport*],[*Contribution*],
  [BRA],[Football] ,[1.25],
  [USA],[Gymnastics] ,[0.51],
  [NED],[Cycling] ,[1.71],
  ),
  caption: none,
)

The coaching contribution rate for both the Brazilian soccer team and the Dutch cycling team is greater than $1$, indicating that the weighted number of medals increased by more than a factor of one after the change point, suggesting that the impact of coaching is significant. In contrast, the U.S. Gymnastics team's coaching contribution rate was $0.51$, which represents a relatively slow increase in weighted medal counts relative to the Brazilian soccer team and the Dutch cycling team, suggesting that the impact of coaching is less significant.

== Choosing to invest in a sport with a “great coach”

=== Expansion of data

To better select investment projects as well as predict contribution rates, we increased the three data sets described above to eight. The added data are as follows:

#figure(
  table(
  columns: (1fr,2fr,2fr,2fr,4fr),
  inset: 3pt,
  stroke:none,
  align: (center,center,center,center,center),
 [*NOC*],[*Sport*],[*ChangeYear*],[*Contribution*],[*GreatCoach*],
[CHN],[Volleyball],[2016],[0.49],[郎平],
[USA],[Gymnastics],[2012],[0.51],[Béla Károlyi],
[NED],[Cycling],[2000],[1.71],[Max van der Stoep],
[CHN],[Table Tennis],[2016],[0.34],[刘国梁],
[FRA],[Fencing],[2000],[0.26],[Pierre Louaillier],
[BRA],[Football],[2020],[1.25],[Tite],
[KEN],[Athletics],[2008],[1.33],[Carlos Lopes],
[GBR],[Swimming],[2016],[9.15],[Daniel Jamieson&Paul Newsome]

  ),
  caption: none,
)

=== Quantifying the level of a country in a given project

Unlike the weighted medal count, we also need to consider the total number of participants. Because a country didn't win a medal in a certain event doesn't mean it doesn't have any competitiveness in that event. So we define the *level* as:

/*
Level=4*Gold_t+3*Silver_t+1*Bronze_t+0.5*No medal_T
*/
$
  "Level" = 4*"Gold"_t + 3*"Silver"_t + 1*"Bronze"_t + 0.5*"No medal"_t
$


Where:
//No medal_T is the number of sports in which the country competed in year t but did not win a medal.
- $"No medal"_t$ is the number of sports in which the country competed in year $t$ but did not win a medal.

We calculate the average *Level* for the five years before the point of change as the historical level of a state in a given program, and obtain the following data:

#figure(
  table(
  columns: (1fr,2fr,2fr),
  inset: 3pt,
  stroke:none,
  align: (center,center,center),
  [*NOC*],[*Sport*],[*Level*],
[CHN],[Volleyball],[19.2],
[USA],[Gymnastics],[59.3],
[NED],[Cycling],[11.5],
[CHN],[Table Tennis],[34.7],
[FRA],[Fencing],[42.3],
[BRA],[Football],[44.9],
[KEN],[Athletics],[33.2],
[GBR],[Swimming],[32.6]
  ),
  caption: none,
)
Averaging this out to $32.6$, we believe that the “great coach” effect is more likely to occur when a country's historical ability in a particular program is around $32.6$.

=== Choose sports that are prone to the “great coach” effect

By counting the total number of medals for Sport in the eight data sets
//Gold_t+Silver_t+Bronze_t,t=change year

$
  "Total" = "Gold"_t + "Silver"_t + "Bronze"_t, t = "Change Year"
$
We get:
#figure(
  table(
  columns: (2fr,3fr),
  inset: 3pt,
  stroke:none,
  align: (center,center),
  [*Sport*],[*Number of medals*],
[Volleyball],[72],
[Gymnastics],[72],
[Cycling],[67],
[Table Tennis],[72],
[Fencing],[67],
[Football],[72],
[Athletics],[72],
[Swimming],[72]
  ),
  caption: none,
)

With an average of $70.75$, we believe that Sports with a total medal count of around $70.75$ are more likely to experience the “great coach” effect. Based on the data from the 2028 Olympic Games, we selected three eligible sports: Artistic Gymnastics (total medals: $67$), Water Polo (total medals: $78$) and Wrestling (total medals: $72$).

===  Selection of countries based on historical level

/*
Define Level_Sport_t1~t2 as t1~t2, the average level of a certain country in a certain program
We calculated each country's Level_Gymnastics_2008~2024, Level_Water Polo_2008~2024, and Level_Wrestling_2008~2024 as their respective historical level in this program. Combining the above data, we have selected the following three countries and programs that are suitable for investing in “great coaches”:

GER Artistic Gymnastics : 29.25
SRB Water Polo : 36.4
JPN Wrestling : 24.7
*/

Define:
 $"Level"_"Sport"_t_1~t_2$ as the average level of a certain country in a certain program from $t_1$ to $t_2$.
 
 We calculated each country's $"Level"_("Gymnastics",(2008~2024))$, $"Level"_("Water Polo",(2008~2024))$, and $"Level"_("Wrestling",(2008~2024))$ as their respective historical level in this program. Combining the above data, we have selected the following three countries and programs that are suitable for investing in “great coaches”:

#figure(
  table(
  columns: (2fr,2fr,2fr),
  inset: 3pt,
  stroke:none,
  align: (center,center,center),
  [*Country*],[*Sport*],[*Level*],
[GER],[Artistic Gymnastics],[29.25],
[SRB],[Water Polo],[36.4],
[JPN],[Wrestling],[ 24.7]
  ),
  caption: none,
)

=== Estimating the contribution of their “great coaches”

The data we have so far is shown below:

#figure(
  table(
  columns: (2fr,3fr,2fr,2fr,3fr),
  inset: 3pt,
  stroke:none,
  align: (center,center,center,center,center),
 [*NOC*],[*Sport*],[*Contribution*],[*Level*],[*Number of medals*],
[CHN],[Volleyball],[0.49],[19.2],[72],
[USA],[Gymnastics],[0.51],[59.3],[72],
[NED],[Cycling],[1.71],[11.5],[67],
[CHN],[Table Tennis],[0.34],[34.7],[72],
[FRA],[Fencing],[0.26],[42.3],[67],
[BRA],[Football],[1.25],[44.9],[72],
[KEN],[Athletics],[1.33],[33.2],[72],
[GBR],[Swimming],[9.15],[32.6],[72],
[GER],[Artistic Gymnastics],[nan],[29.25],[67],
[SRB],[Water Polo],[nan],[36.4],[78],
[JPN],[Wrestling],[nan],[24.7],[72]

  ),
  caption: none,
)

It can be seen that when the “Great Coach” effect occurs, Number of medals are all above and below $70.75$, with an extreme variance of 5. Let's try to build a multiple linear regression model：

//Contribution=β_0+β_1*Level+β_1*Number of medals

$
  "Contribution" = beta_0 + beta_1*"Level" + beta_2*"Number of medals"
$

The model can be solved using the least squares method from the Statsmodels library in Python and the results obtained are shown below:

//Figure4 OLS Regression Results.png
#figure(
  image("Figure4 OLS Regression Results.png", width: 70%),
  caption: [OLS Regression Results.png],
)

From the results, the *t-test* result of *Number of medals* is not satisfactory, and the t-statistic corresponding to *Number of medals* has a P-value of 0.599, which shows that the linear relationship between *Number of medals* and Contribution is not significant. Combined with the previous results that the extreme deviation of *Number of medals* (extreme deviation = $5$) is small, we believe that when the “great coach” effect occurs, the influence of *Number of medals* on the contribution of coaches is very small. The t-statistic corresponding to *Level* corresponds to a *P-value* of $0.656$, and we get The linear relationship between *Level* and *Contribution* is not significant.

We conjecture that *probably* the country's technological level as well as economic level will amplify or attenuate the “great coaches” effect. This is because upgrading high-quality professional training facilities can help a coach achieve the training effect he wants. The coach can also use science and technology to analyze more appropriate training methods and research more effective countermeasures, thus increasing the number of medals. We are limited by the small amount of data available for the “Great Coach” effect, as well as the requirement of the question that our model and data analysis must use only the data set provided. Therefore, we can only predict a vague coaching contribution rate by comparing the data, and the prediction is as follows:

#figure(
  table(
  columns: (2fr,4fr,3fr,3fr,4fr),
  inset: 3pt,
  stroke:none,
  align: (center,left,center),
  [*NOC*],[*Sport*],[*Contribution*],[*Level*],[*Number of medals*],
[GER],[Artistic Gymnastics],[0.7],[29.25],[67],
[SRB],[Water Polo],[1],[36.4],[78],
[JPN],[Wrestling],[0.5],[24.7],[72]

  ),
  caption: none,
)

So we suggest that *Germany* might consider investing in “Great Coaches” on *Artistic Gymnastics* with an expected coaching contribution of $0.7$, *Serbia* might consider investing in “Great Coaches” on* Water Polo* with an expected coaching contribution of $1$, and *Japan* might consider investing in “Great Coaches” on *Wrestling* with an expected coaching contribution of $0.5$.

#pagebreak()

= Task 5: Analysis of other factors.

// 我们在讨论”伟大教练“效应时，分析了国家的经济水平可能会增强或减弱”伟大教练“效应。于是我们猜想当”伟大教练“效应不存在时，经济水平也会影响奖牌数。下面是中国队在从1984年到2024年的总奖牌数趋势图：

When discussing the "great coach" effect, we analyzed that the country's economic level may enhance or weaken the "great coach" effect. So we conjectured that when the "great coach" effect does not exist, the economic level will also affect the number of medals. The following is the trend chart of the total number of medals won by the Chinese team from 1984 to 2024:

#figure(
  image("Figure5 CHN Total Medals.png", width: 80%),
  caption: [China Medal Trend],
)
/*

1984年到2024年中国的经济水平在不断提升。图中剔除掉2008年北京奥运会的数据（为了排除东道主的影响），我们发现中国的总奖牌数也呈上升的趋势。到这里我们认为经济水平会影响奥运奖牌数。

下面是古巴队在从1984年到2024年的总奖牌数趋势图：
*/

From 1984 to 2024, China's economic level has been continuously improving. By excluding the data from the 2008 Beijing Olympics (to eliminate the impact of the host country), we found that China's total medal count is also on the rise. At this point, we believe that the economic level will affect the number of Olympic medals.

The following is the trend chart of the total number of medals won by the Cuban team from 1984 to 2024:

#figure(
  image("Figure6 CUB Total Medals.png", width: 80%),
  caption: [Cuba Medal Trend],
)

/*

古巴在1996年至2020年期间，经济总体呈现增长趋势，但是总奖牌数却有下降的趋势。我们发现古巴的人口数量远小于中国的，且古巴存在体育人才流失的现象，因此，我们认为人口数量也会影响奥运奖牌数。

下面是印度队在从1984年到2024年的总奖牌数趋势图：
Figure7 IND Total Medals

印度在2000年至2020年期间，经济快速增长，成为全球经济增长最快的国家之一。且印度的人口数量相对较多，但是总奖牌数也没有表现出明显的上升趋势。我们猜测可能是国家的基础设施不足和经济分配问题导致人口数量转化为运动员的比例较小，从而影响总奖牌数的变化。

由此我们得到，经济水平，人口数量，基础设施水平会在一定程度上影响奖牌数的变化。
*/

From 1996 to 2020, Cuba's economy showed an overall growth trend, but the total number of medals showed a downward trend. We found that Cuba's population is much smaller than China's, and Cuba has a phenomenon of talent loss in sports. Therefore, we believe that population size will also affect the number of Olympic medals.

The following is the trend chart of the total number of medals won by the Indian team from 1984 to 2024:

#figure(
  image("Figure7 IND Total Medals.png", width: 80%),
  caption: [India Medal Trend],
)

From 2000 to 2020, India's economy grew rapidly, becoming one of the fastest-growing economies in the world. India also has a relatively large population, but the total number of medals has not shown a clear upward trend. We speculate that it may be due to insufficient national infrastructure and economic distribution issues that the proportion of the population converted into athletes is relatively small, thereby affecting the change in the total number of medals.

From this, we conclude that the* economic level*, *population size*, and *infrastructure level* will affect the change in the number of medals to a certain extent.
