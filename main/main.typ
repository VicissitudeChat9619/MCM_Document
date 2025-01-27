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


== Literature Review


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
  image("C_2data initialization.png", width: 40%),
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
J(Theta) = -1/n sum_(i=1)^n [y^{(i)} log(hat(y)^(i)) + (1 - hat(y)^(i)) log(1 - y^{(i)})]
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

= Task 4: Analysis of the "Great Coach" Effect

== Main tasks

We used mathematical modeling to determine the point when a coach came on board and quantify the contribution of coach turnover to the medal count based on medal data `summerOly_athletes.csv` for Olympic athletes through a change point detection method. We then recorded the country's level before the point of change in a given sport (the detailed quantification process is written in #link(<Quantifying>)[Quantifying]). By analyzing the historical level as well as the total number of medals in the historical sport, three countries were selected to find the sports in which they should consider investing in “Great Coach” and to estimate their impact.

// 我们通过数学建模，基于奥运会运动员的奖牌数据（summerOly_athletes.csv），通过变化点检测方法确定教练上任的时间点和量化教练更替对奖牌数的贡献。然后我们记录下该国家在某个项目的变化点前的水平（详细的量化过程写在了2.5.2），通过分析历史水平以及历史项目的总奖牌数，选择出三个国家，找到它们应考虑投资于 “优秀 ”教练的运动项目，并估计其影响。

== Data preprocessing

We take the medal count (gold, silver, bronze) of a country in a given sport at each Olympics. Because the types of medals (gold, silver, bronze) reflect different variations in the strength of a country in a particular sport, we assign different weights to each type of medal: Gold: 3; Silver: 1; Bronze: 0.5. then: 

$
  WW_t=3*"Gold"_t+1*"Silver"_t+1*"Bronze"_t
$

Where:
+ $WW_t$ is the weighted total number of medals in a sport for that country in year $t$.
+ $"Bronze"_t$ is the number of bronze medals won by the country in a sport in year $t$.
+ $"Silver"_t$ is the number of silver medals won by the country in a sport in year $t$.
+ $"Gold"_t$ is the number of gold medals won by the country in a sport in year $t$.

== Bayesian change point detection

=== Principle

*Bayesian change point detection* is a method based on Bayesian statistics for identifying change points in time series data. A change point is a point where there is a significant change in the distribution of the data, such as a change in the mean, variance, or trend. Bayesian methods can deal with the change point detection problem flexibly by introducing prior and posterior distributions and providing uncertainty estimates of the location of change points.//@5-1

=== Model Assumption

Assume that the weighted medal count $WW_t$ in year $t$ is distributed before and after the change point obeys a normal distribution with different means and variances:



=== Quantifying the level of a country in a given project <Quantifying>