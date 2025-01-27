#let fontsize = 12pt
#let title-fontsize = 16pt
#let fonttype = "Times New Roman"

#let page_counter = counter(page)
#page_counter.update(0)
#let first-line-indent = 20pt
// #set par(
//     // first-line-indent: first-line-indent,
//     hanging-indent: -first-line-indent,
//     linebreaks: "optimized",
  )
#pad(left:-first-line-indent)[#line(length: 100%, stroke: black)]

#set page(
header: pad(left:-first-line-indent)[#box(stroke: (bottom: 1pt), inset: 5pt)[
    #text("Team #")
    #text("2518527")
    #h(1fr)
    #text("Appendices Report Page")
    #page_counter.display("1 of 1", both: true)
]],
header-ascent: 20%,
margin: (left: 80pt, right: 50pt, top: 60pt, bottom: 40pt),
)

= Report on Use of AI


+ *Deepseek `Deepseek` (Open score version,Deepseek-R1-665B)*

*Queary 1: *

```plaintext
Please translate the following markdown content into a typst code.
== 3.1 Model Construction
Attempt to Build a Neural Network Model 

A neural network model is constructed to fit the characteristics of countries that have won their first medal in past years.

Let $X=[x_1,x_2,x_3]$ Here ，
| Symbols | meaning                          |
|---------|----------------------------------|
| $x_1$   | The number of editions in which the country has participated without winning any medals since its first participation. |
| $x_2$   | The number of athletes the country has in this edition        |
| $x_3$   | The average number of medals awarded in this edition.               |


Use $X$ as the input layer, establish two hidden layers with 4 and 3 neurons respectively, and set the output layer with 1 neuron to represent the probability of winning a medal.

$y$ represents whether a medal is won in the next edition, where $y=0$ indicates no medal,and $y=1$ indicates a medal is won.

== 3.2 Model Solution

=== 3.2.1 Data Preparation
The dataset is processed based on the files provided on the official website. The samples are split into training and testing sets with a ratio of 0.2.

The flowchart of sample data preparation is represented as follows.

![C_2data initialization.png](<attachment:C_2data initialization.png>)

=== 3.2.2 Neural Network Framework Construction
$a^{(i)}$ represents the activation value of the i-th layer.

$a^{(2)}=g(\Theta^{(1)}X)$

$a^{(3)}=g(\Theta^{(2)}a^{(2)})$

$a^{(4)}=g(\Theta^{(3)}a^{(3)})$

$\hat y=a^{(4)}$

Here, $\Theta^{(i)}$ represents the propagation matrix, and $g$ denotes the activation function. In this case, the sigmoid function is used as the activation function, with an output range of $(0,1)$

=== 3.2.2 Cost Function

The binary cross-entropy loss function measures the difference between the predicted value $\hat y$ and the actual value $y$ .


$$
J(\Theta)=-\frac{1}{n}\sum^n_{i=1}[y^{(i)}log(\hat y^{(i)})+(1-\hat y^{i})]
$$

=== 3.2.3 Problem Transformation

The model's learning involves minimizing the loss function by optimizing parameters through sample-based learning.

=== 3.2.4 Parameter Optimization
==== 3.2.4.1 Optimization Methods
Use the gradient descent method to optimize the loss function. By iterating over each sample, calculate the gradients using backpropagation to obtain $\frac{\partial J}{\partial \Theta^{(i)}_{jk}}$
Let $\eta$ learning rate.

The parameter update rule is given by:
$$
\Theta^{(i)}_{jk}:= \Theta^{(i)}_{jk}-\eta \frac{\partial J}{\partial \Theta^{(i)}_{jk}}
$$
This process is repeated iteratively until convergence or the maximum number of iterations is reached.

Since gradient descent requires repeated calculations over a large number of samples to compute the average, to improve algorithm performance, we adopt a grouped averaging approach for descent.Let$m$represent the number of groups into which the samples will be divided. In each iteration, every group is processed, and the descent is performed using the average gradient of each group.

==== 3.2.4.2 Model Training
The main steps are shown in the following:

**step1**:Initialize the parameters $\Theta^{(1)}, \Theta^{(2)}, \Theta^{(3)}$ the maximum number of iterations (epochs), and the number $m$ of groups. 

**step2**:Forward propagation to compute $a^{(2)},a^{(3)},a^{(4)}$

**step3**:Group the samples based on  $m$ , which represents the number of groups

**step4**: For each sample group, calculate the gradient of each parameter using the backpropagation method.

**step5**: Perform gradient descent for each parameter after obtaining the average gradient of each sample group, and repeat until all groups have completed their iterations.

**step6**: Record the loss function value after each iteration.

**step7**: Determine whether the maximum number of iterations has been reached. If satisfied, stop the training; otherwise, repeat steps 2 to 7.

== 3.3 Model evaluation
Through the trained model, the learning curve and accuracy curve are observed, and the fitting performance is as follows:

![C_2_loss_Accuracy.png](attachment:C_2_loss_Accuracy.png)

It can be observed that the classifier trained by this model demonstrates high accuracy and a low loss value.

The final model predicts the probability of countries that have not previously won a medal winning a medal in the next event. The results are shown in the figure below. ***Since many countries have a probability close to zero, only the top 10 countries most likely to win their first medal are displayed.***

![The top 10 countries most probable to win first medal.png](<attachment:The top 10 countries most probable to win first medal.png>)
```

*Output 1: *

```plaintext
== Model Construction

Attempt to Build a Neural Network Model

A neural network model is constructed to fit the characteristics of countries that have won their first medal in past years.

Let $X = [x_1, x_2, x_3]$. Here:
#figure(
  table(
    columns: (1.2fr, 4fr),
    inset: 4pt,
    stroke: none,
    align: (center, left),
    [*Symbols*], [*Meaning*],
    [$x_1$], [The number of editions in which the country has participated without winning any medals since its first participation.],
    [$x_2$], [The number of athletes the country has in this edition],
    [$x_3$], [The average number of medals awarded in this edition.]
  ),
  caption: [Variable Definitions],
)<tab-symbols>

Use $X$ as the input layer, establish two hidden layers with 4 and 3 neurons respectively, and set the output layer with 1 neuron to represent the probability of winning a medal.

Define $y$ as the medal-winning indicator:
$y = cases(
  0 quad "indicates no medal",
  1 quad "indicates a medal is won"
)$

== Model Solution

=== Data Preparation
The dataset is processed based on the files provided on the official website. The samples are split into training and testing sets with a ratio of 0.2.

#figure(
  image("C_2data_initialization.png", width: 80%),
  caption: [Data Preparation Flowchart],
)<fig-flow>

=== Neural Network Framework Construction
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
The binary cross-entropy loss function measures the difference between predicted and actual values:

$
J(Theta) = -1/n sum_(i=1)^n [y^{(i)} log(hat(y)^(i)) + (1 - hat(y)^(i)) log(1 - y^{(i)})]
$

=== Problem Transformation
The model's learning involves minimizing the loss function by optimizing parameters through sample-based learning.

=== Parameter Optimization
==== Optimization Methods
Use gradient descent method with backpropagation:

Update rule:
$
Theta_(jk)^(i) := Theta_(jk)^(i) - eta (del J)/(del Theta_(jk)^(i))
$
With grouped averaging approach:
- Let $m$ = number of sample groups
- Process each group's average gradient per iteration

==== Model Training
**Training Procedure**:
1. Initialize parameters $Theta^(1)`, $Theta^(2)`, $Theta^(3)` and hyperparameters
2. Forward propagation: Compute $a^(2)`, $a^(3)`, $a^(4)`
3. Split samples into $m$ groups
4. Backpropagation: Calculate gradients per group
5. Update parameters using group averages
6. Record loss values
7. Repeat until convergence or max epochs

== Model Evaluation
#figure(
  image("C_2_loss_Accuracy.png", width: 90%),
  caption: [Learning Curve and Accuracy],
)<fig-eval>

#figure(
  image("top_10_countries.png", width: 100%),
  caption: [Top 10 Potential First-time Medal Winners],
)<fig-results>

[*Note*]: Only displaying countries with probability > 0.1 for clarity.
```

+ *Github `Copilot` (January 17, 2025, `Release`, Copilot-v1.0)*

Auto-completions for code used in cleaning and preprocessing data for Task*.1*,*.2* models.


+ *OpenAI `ChatGPT` (January 17, 2025, `Release`, GPT-o1-min)*

*Queary 1: *

```plaintext
Please help me find ten examples of great coaches who can greatly impact the number of Olympic medals.
```

*Output 1: *

```plaintext
1. Lang Ping and the Chinese Women's Volleyball Team
Coach: Lang Ping
Sport: Volleyball
Coaching Period: 1995-1999, 2013-2021
Achievements:
During her first tenure from 1995 to 1999, Lang Ping led the Chinese women's volleyball team out of a trough, winning the silver medal at the 1996 Atlanta Olympics and the silver medal at the 1998 World Championships.
From 2013 to 2021, Lang Ping returned to coach the Chinese women's volleyball team, leading the team to win the 2015 World Cup championship, the 2016 Rio Olympics gold medal, and the 2019 World Cup championship.
Impact: Lang Ping's coaching not only enhanced the team's technical and tactical level but also strengthened the players' self-confidence and team cohesion. Her training methods and management style significantly improved the competitiveness of the Chinese women's volleyball team on the international stage.

2. Béla Károlyi and the Romanian and American Women's Gymnastics Teams
Coach: Béla Károlyi
Sport: Gymnastics
Coaching Period: Late 1970s to 1980s (Romania), 1999 to early 2000s (USA)
Achievements:
Romanian Gymnastics Team: Under the coaching of Béla Károlyi, the Romanian gymnastics team became a world gymnastics powerhouse. At the 1980 Moscow Olympics, the Romanian gymnastics team won the team gold medal, and Nadia Comăneci became the first gymnast to score a perfect 10 at the Olympics.
American Gymnastics Team: After joining the American gymnastics team in 1999, he introduced Romanian training methods, quickly improving the overall level of the American gymnastics team. At the 2000 Sydney Olympics, the American women's gymnastics team won the team silver medal; at the 2012 London Olympics, the American women's gymnastics team won the team gold medal, and Gabby Douglas won the all-around gold medal.
Impact: Béla Károlyi's strict training methods and technological innovations greatly improved the athletes' technical level and competitive performance. His coaching style and training system played a key role in the rise of the Romanian and American gymnastics teams.
Max van der Stoep and the Dutch Cycling Team

3.Coach: Max van der Stoep
Sport: Cycling
Coaching Period: 2000-2008
Achievements:
Under the coaching of Max van der Stoep, the Dutch cycling team performed outstandingly at the 2004 Athens Olympics and the 2008 Beijing Olympics, winning multiple medals. The Dutch team made significant progress in track cycling and became one of the world's top teams in the sport.
Impact: Van der Stoep introduced new training methods and tactical systems, improving the athletes' physical fitness and technical level. His coaching not only enhanced the competitive level of the Dutch cycling team but also promoted the overall development of Dutch cycling.

4. He Zhenliang and the Chinese Table Tennis Team
Coach: He Zhenliang
Sport: Table Tennis
Coaching Period: 1980s
Achievements:
During the 1980s, He Zhenliang served as the head coach of the Chinese table tennis team, leading the team to achieve excellent results in multiple World Championships and Olympics, establishing the dominant position of the Chinese table tennis team in the world table tennis arena.
Impact: Zhenliang's coaching style and training methods greatly improved the overall level of the Chinese table tennis team, cultivating many world champions and Olympic champions, such as Deng Yaping and Qiao Hong.

5. Fabio Capello and the Italian and English Football Teams
Coach: Fabio Capello
Sport: Football
Coaching Period: 1990s (Italy), 2008-2011 (England)
Achievements:
Italian Football Team: During the 1990s, Capello coached AC Milan and Juventus, leading the teams to win multiple Serie A league championships and UEFA Champions League titles.
English Football Team: From 2008 to 2011, Capello coached the England national team, leading the team to achieve certain results at the 2010 World Cup.
Impact: Capello's tactical system and management style greatly improved the competitive level of the teams, cultivating many world-class players, such as Paolo Maldini and Francesco Totti.

6. Pierre Louaillier and the French Fencing Team
Coach: Pierre Louaillier
Sport: Fencing
Coaching Period: 2000-2008
Achievements:
Under the coaching of Pierre Louaillier, the French fencing team performed outstandingly at the 2004 Athens Olympics and the 2008 Beijing Olympics, winning multiple medals. The French team made significant progress in fencing and became one of the world's top teams in the sport.
Impact: Louaillier introduced new training methods and tactical systems, improving the athletes' technical level and competitive performance. His coaching not only enhanced the competitive level of the French fencing team but also promoted the overall development of French fencing.

7. Igor Kovalyov and the Russian Figure Skating Team
Coach: Igor Kovalyov
Sport: Figure Skating
Coaching Period: 1990s-early 2000s
Achievements:
Under the coaching of Igor Kovalyov, the Russian figure skating team won multiple World Championships and Olympic gold medals in the 1990s and early 2000s.
Impact: Kovalyov's training methods and technological innovations greatly improved the athletes' technical level and artistic expression. His coaching style and training system played a key role in the rise of the Russian figure skating team, cultivating many world champions and Olympic champions, such as Irina Slutskaya and Maria Butyrskaya.

8. Joe Mazurkowski and the Polish Volleyball Team
Coach: Joe Mazurkowski
Sport: Volleyball
Coaching Period: 1980s-1990s
Achievements:
Under the coaching of Joe Mazurkowski, the Polish volleyball team won multiple European Championship and World Championship medals in the 1980s and 1990s.
Impact: Mazurkowski's training methods and tactical systems greatly improved the overall level of the Polish volleyball team, cultivating many world-class players, such as Tomasz Zaremba and Marek Sadowski. His coaching not only enhanced the competitive level of the Polish volleyball team but also promoted the overall development of Polish volleyball.

Summary
These "great coaches" have significantly improved the performance of athletes and teams by introducing advanced training methods, strict management, and technological innovations. Their success is not only reflected in the increase in the number of medals but also in the cultivation of many world champions and Olympic champions, which has had a profound impact on the development of sports projects. These examples fully demonstrate the important role of coaches in sports competitions and provide valuable references for the recruitment of coaches and the formulation of sports policies by national Olympic committees.
```
