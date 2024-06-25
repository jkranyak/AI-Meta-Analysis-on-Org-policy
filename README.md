# AI & Organizational Strategy Meta Analysis 
# All code written by Jesse Kranyak, 
### Code edits made with chatgpt4omni, llama3-7b 
### Agent Code inspired by David Andrej 






# Impact of Artificial Intelligence Organizational Strategies and Policy:
## A Comparative Review Between Artificial Intelligence and Human Researcher 
on 
### the Current Academic Literature 
and 
### Meta Analysis to Determine Primary Qualitative Themes. 

# Introduction
This research seeks to explore significant trends in organizational strategy and policy via a comprehensive meta-analysis, viewed through multiple lenses. This exploration will be multi-modal; exploring machine learning, neural networks and artificial intelligence, as well as that of a human researcher. A collection of articles and studies will be collected based on parameters outlined in methods to construct a substantial framework reflecting our present understanding of operational systems and how they are being impacted by artificial intelligence. First this data will be categorized in a meta analysis to determine the qualitative themes by a human researcher, the author, and then through a machine learning tool, followed by a neural network method of exploratory analysis and finally an AI set of Agent ‘Researchers’.  Along each step of the process we will stop and provide a visual analysis of the data and explore the themes derived from each method.  

For our multi modal approach to the meta analysis we will explore several types of data extraction processes. The first will be the human researcher, before any work is done on the digital side of this research, the papers will need to be classified into their thematic elements.  It is the intention of this process to remove bias that would come from the alternative modes of grouping that follow.  Following the human we will start with the least complex and move up in model complexity with each step.  Next we will use a machine learning model known as Latent Dirichlet Allocation (LDA) to analyze the data, which will search for hidden patterns and word clusters in our documents to find themes.  Following that we will look at themes produced by TensorFlow and PyTorch to produce an autoencoder, a type of neural network that learns to compress and then reconstruct data, making it useful for reducing data size and identifying patterns. Our final iteration of processing the data and extracting themes will be done using an agent team built on the API of either GPT4 Turbo or Llama3 70b.  This final representation is one of the more advanced applications of artificial intelligent networks available. In this iteration a team of artificial intelligence including a ‘thematic analyst’ and a ‘senior business developer’ will be built to work together under a ‘manager’ to classify the papers into thematic elements.  

In conclusion, this article serves multiple purposes; first to discuss the themes of AI’s impact on organizational strategy and policy across the globe, and further as a basis for research that aims to determine if there is latent data hidden inside of this meta analysis. The final goal of this papers result section will be to reveal a more intricate and interconnected network within the global organizational strategy and policy ecosystem, by combining all of these different modes of research together.  It would be the expected outcome that while we discover some gaps between the types of exploration and analysis we should end up with a much more robust analysis of the research topic in general. 




# Literature Review

Articles were sought out using multiple search methods outlined in supplemental 1 of this paper. After collecting 52 papers in total the abstracts were assessed and narrowed down to a final list of  26 articles that were reviewed and analyzed using multiple processes to obtain thematic elements. These include government documents, articles from global organizations and research articles.  Once the list of articles was satisfactory in terms of context related to the topic of AI in organizational strategy & policy, a spreadsheet of the articles was created containing; article title,  keywords , author, abstract, the full article itself, source, year, and web link to article. In this manner the spreadsheet was used to keep articles organized as well as used in the next step of process which is machine analysis. 

# Methodology

Multiple approaches were taken in our methodology as part of this study.  The first was all manual and included no software.  Articles were read and keywords were identified.  A list of keywords was started for each article and placed in a new column available in the accompanying git repository file ‘AI_Meta_analysis.csv’.  The keyword groups were analyzed and separated into groups to create our first set of manually created thematics.  Next, a new spreadsheet was created containing only keywords, and sorted alphabetically. A period of two weeks was taken away from the articles to try and remove categorical biases.  These keywords were then analyzed outside of the context of the article titles and article thematics, and systematically reduced by combining similar orientation, until there was a sufficient enough reduction where it did not make objective rational to continue creating categories; thus creating a second set of manually derived thematics. Finally, after these two groups of thematic elements were constructed, a comparative analysis was conducted to see if the groups shared relevant convergent or divergence and a final thematic list was obtained.  This unique structure of methodology was created to attempt an unbiased collection of literature thematics, allowing for a robust meta-analysis of the impact of AI on organizational strategies and policies.

Next, the process of thematic extraction by machine learning processes was started. All articles were loaded into a custom built python pipeline using Google Colab, where a series of programming functions performed both Latent Dirichlet Allocation (LDA) and Non-Negative Matrix Factorization (NMF). In order to feed our articles into both of these algorithm based systems the articles are first broken down into individual words and all filler language is removed, this process is explained in greater detail in the program itself on the repository.  Briefly, this process is done by converting all of our papers into a Document-Term Matrix (DTM).  The DTM is a matrix used in natural language processing (NLP) to represent the frequency of terms (words, or tokens) occurring in our collection of articles. In the DTM, each row corresponds to a document and each column corresponds to a term (Priya, 2022). The value in a cell represents the frequency of the term in the document.  The next step in this process is to convert the DTM into a Gensim corpus.  This corpus is a list of lists, with each inner list representing a document as tuples of term ids and their respective frequencies. This format is suitable for Gensim's topic modeling algorithms which we will then use the methods described earlier, LDA and NMF.

One issue presented here is that we will need to input the number of thematic elements prior to running the Gensim modeling techniques (Shristikotaiah, 2022).  In order to do the best number of topics needs to be estimated based on the ‘coherence of topic models’ with different numbers of topics.  One of the first blocks of code in the repository will address this by compiling a report on the coherence, and define a function called `compute_coherence_values`. This function iterates over a specified range of topic numbers, fits an LDA model for each number, and calculates the coherence score, which measures the model's interpretability and semantic consistency (Bellaouar, Bellaouar & Ghada, 2021). This helps in selecting the optimal number of topics for the model.  

Our final AI process will require the building of multiple agent teams.  In order to create the best agents for this job assigning the most efficient jobs will be of the utmost importance and consulting another large language model will assist in creating an AI team and developing our code as well as work flow.  Some tricks to expedite the efficiency of the process will be to assign directly the authors goals and position with that of the AI Agents, for instance, we can assign the final agent the role of: “researcher trying to obtain a Masters degree in an MBA, working on a meta-analysis in our topic”.  This type of prompt engineering can have a positive effect on the outcome of our research. It is important to note at this point that this aspect of the project has the most variability, so for the sake of brevity we will keep our prompt short.  The choice of not using an AI such as ChatGPT and instead using a team of AI Agents on a ChatGPT API in a project like this has many advantages.  Essentially, when arranged in an agent framework giving the AI smaller jobs and having it work with other versions of itself, is more efficient than asking a large language model one question and receiving one answer.  Each agent in the team can be specialized in different aspects of the meta-analysis process, enhancing the overall quality and efficiency of the work. One agent will focus on gathering and summarizing relevant literature, ensuring a comprehensive and up-to-date review of existing research. Another agent will specialize in data extraction and synthesis, meticulously organizing findings and identifying key themes across studies. A third agent can handle statistical analysis, applying advanced techniques to combine results and draw meaningful conclusions. Additionally, an agent dedicated to writing and structuring the report can ensure that the final document is coherent, well-organized, and adheres to academic standards. By leveraging the specialized skills of each agent, the team approach not only speeds up the process but also improves the accuracy, depth, and overall quality of the meta-analysis.  The full version of this paper would exceed 100 pages, so again continue to excuse the brevity in these sectors, we will leave off the bulk of the AI’s output, but it is available in the Supplemental 3 & 4 and the accompanying git repository at https://github.com/jkranyak/705_MBA_FinalPaper/blob/main/705_MBA_FinalPaper.ipynb.  

# Research Question & Hypothesis
Research Question
What are the primary qualitative themes in organizational strategy and policy development as determined by a comparative meta-analysis using human researchers, machine learning models, neural networks, and AI agent teams?
Hypotheses
Hypothesis 1: Thematic analysis conducted by artificial intelligence models will identify more nuanced and diverse themes in organizational strategy and policy development than those identified by a human researcher.
Hypothesis 2: Machine learning models, such as Latent Dirichlet Allocation (LDA), will uncover hidden patterns and word clusters that a human researcher might overlook in the academic literature.
Hypothesis 3: Neural network approaches, particularly auto-encoders, will effectively reduce data size while preserving critical thematic elements, providing a comprehensive overview of organizational strategies and policies.
Hypothesis 4: AI agent teams, utilizing advanced models like GPT-4 Turbo or Llama3 70b, will outperform both human researchers and simpler machine learning models in identifying key themes and generating actionable insights for strategic decision-making and policy development.

# Importance of the meta-analysis & experiment

The meta-analysis and AI experiment hold importance in understanding AI's impact on organizational strategies and policies. By synthesizing findings from multiple studies, the meta-analysis provides a comprehensive view, ensuring conclusions are based on a broad spectrum of research. This approach enhances the reliability and validity of results, allowing for the identification of patterns and trends that a single study may miss. The involvement of multiple approaches mitigates biases inherent in any single method, presenting a balanced and objective analysis. 

The experiment, on the other hand, demonstrates the practical application of advanced analytical techniques, showcasing the potential of machine learning, neural networks, and AI agent teams to enhance traditional research methods. By benchmarking the capabilities of AI against human researchers, the experiment provides insights into the strengths and limitations of AI in qualitative research. These findings offer actionable insight for organizational leaders and policymakers, helping them strategize AI adoption and policy development. Additionally, experiments like this can highlight future research directions, guiding the refinement of AI models for better integration into strategic decision-making processes. 

# Results

Manual Results 
Manual research uncovered the following groupings of thematic elements.  First, by reading the articles and assigning keywords to them based on subjective reasoning of this papers author, it was determined that these articles fell into one of seven categories: 1. Business Strategy, market research & competition 
2. Decision making processes
3. Ethical decision making processes 
4. Organizational analysis 
5. Culture and Human Resource Development
6. Regulatory processes 
7. Risk management 
These keywords were then re-organized into groups depending on the theme of the word itself, first by grouping like words together; examples “ethical” and “ethics” and “ethical decision making”.  The Keyword grouping analysis led to a slightly different thematic analysis of the papers, with with predominant themes listed below and seen in fig 1.1; 
Decision making
Ethics oriented
Strategy formation 
Safety and security
Competitive advantage 
Automation and technical advances 
Governance, policy and regulation 
Human Resources
From this comparison, we can see that while the groupings identified through keyword analysis align closely with those identified by manual research, there are slight variations in categorization. For instance, “Organizational Analysis” from the manual research grouping appears to correspond to both “Strategy Formation” and potentially other themes, perhaps having the most potential relating to competition. Others, such as “Automation and Technical Advances” in the keyword analysis, also point to multiple themes, but would similarity rest in competition. Again, “Business Strategy, Market Research & Competition” aligns well with “Competitive Advantage,” but aspects of it may also relate to “Strategy Formation.”

The comparison indicates that while the overall themes are consistent, the method of grouping—whether by human interpretation or keyword analysis—can influence the final categorization, highlighting the inherent bias and subjectivity involved in thematic analysis. Not to mention many of the thematics here obviously align very well with our search. Taking the two lists and combining them into a final list of thematic elements we can narrow this down to their “Thematic Element Codes”.  Each number below will represent a code on the spreadsheet for the themes encountered in each article according to both types of analysis:
Decision Making
   - Decision making
   - Decision making processes
Ethics
   - Ethics oriented
   - Ethical decision making processes
Strategy and Competition
   - Strategy formation
   - Business Strategy, market research & competition
   - Automation and Technical Advances
  - Competitive Advantage
  - Organizational Analysis
Safety, Security, and Risk Management
   - Safety and security
   - Risk management
Human Resources and Culture
   - Human Resources
   - Culture and Human Resource Development
Governance and Regulation:
   - Governance, policy and regulation
   - Regulatory processes
Using these six themes we can apply a number to each article as a code. For instance, article 1 has the assigned theme of “Business Strategy, Market Research & Competition” and the keywords “Decision making, global competition, competitive advantage”; therefore the article can be assigned a 1 and a 3, as the reader assigned focus area falls under code 3 and the keywords support both 3 and elements of decision making processes as well.  Below is a graph of occurrences of themes by code in our analyses:


In this process we can see that the category of ‘Strategy & Competition’ would occur in roughly 9 out of our 27 articles or have a 33% persistence.  


# Results of LDA
(For the full version of this section, it may be best to refer to the GitHub repository here 
https://github.com/jkranyak/705_MBA_FinalPaper/blob/main/705_MBA_FinalPaper.ipynb)

 
As discussed earlier it was necessary to run an optimal coherence model selection to find out how many themes we needed to set our algorithms up to extract.  The result of our optimal model selection is 2 categories, however, the topic of 6 categories has the second highest coherence score. In this case we will run a 2 and a 6 topic analysis. These are the coherence scores:  Num Topics = 2  has Coherence Value of 0.7361
Num Topics = 3  has Coherence Value of 0.6758
Num Topics = 4  has Coherence Value of 0.7072
Num Topics = 5  has Coherence Value of 0.7006
Num Topics = 6  has Coherence Value of 0.7147
Num Topics = 7  has Coherence Value of 0.6911
Num Topics = 8  has Coherence Value of 0.7051
Num Topics = 9  has Coherence Value of 0.7034

For the sake of brevity, we will only run the 6 theme method, running 2 themes yielded themes that were too complex, or rather too overarching, to draw conclusions from.  The six topics selected by LDA are (listed 0-5 as opposed to 1-6); 
 Topic #0:
['bias', 'accountability', 'issue', 'regulatory', 'sector', 'framework', 'transparency', 'risk', 'model', 'financial']
Topic #1:
['industry', 'challenge', 'analysis', 'integration', 'model', 'role', 'sustainable', 'development', 'human', 'risk']
Topic #2:
['sector', 'state', 'new', 'standard', 'approach', 'rd', 'national', 'development', 'public', 'government']
Topic #3:
['definition', 'researcher', 'category', 'framework', 'article', 'machine', 'group', 'application', 'making', 'human']
Topic #4:
['activity', 'work', 'theme', 'value', 'employee', 'legal', 'example', 'culture', 'initiative', 'change']
Topic #5:
['digital', 'relationship', 'new', 'resource', 'innovation', 'human', 'job', 'worker', 'performance', ‘firm’]

We can visualize these by their appearance in words clouds as such: 
Looking at these topics, we can asses the following groups by feeding the categorical lists back into Python programming workflow and have an agent assist us in determining the categorical breakdown of the LDA keywords.  The result of that analysis breaks down to the following analysis using AI Agent translation (all italicized wording below is written by AI):
Regulatory and Ethical Issues
- Keywords: bias, accountability, issue, regulatory, sector, framework, transparency, risk, model, financial.
Industry Challenges and Integration
- Keywords: industry, challenge, analysis, integration, model, role, sustainable, development, human, risk.
National and Public Sector Approaches
- Keywords: sector, state, new, standard, approach, rd, national, development, public, government.
Research Frameworks and Human Factors
- Keywords: definition, researcher, category, framework, article, machine, group, application, making, human.
Organizational Culture and Change
- Keywords: activity, work, theme, value, employee, legal, example, culture, initiative, change.
Digital Innovation and Workforce
Keywords: digital, relationship, new, resource, innovation, human, job, worker, performance, firm.


# Results of NMF
This results in the following outcome:
Topic #0:
['human', 'developing', 'partnership', 'model', 'needed', 'federal', 'government', 'standard', 'development', 'rd']
Topic #1:
['finding', 'model', 'capability', 'financial', 'operational', 'relationship', 'resource', 'innovation', 'firm', 'performance']
Topic #2:
['percent', 'warehousing', 'new', 'hiring', 'algorithm', 'labor', 'task', 'job', 'firm', 'worker']
Topic #3:
['assessment', 'capability', 'approach', 'effectiveness', 'financial', 'model', 'analysis', 'image', 'human', 'risk']
Topic #4:
['researcher', 'computer', 'group', 'category', 'framework', 'article', 'machine', 'application', 'making', 'human']
Topic #5:
['new', 'industry', 'initiative', 'challenge', 'sustainability', 'role', 'sustainable', 'model', 'development', ‘change']

Based on the keywords from the NMF analysis, here is a breakdown of the topics and word clouds for the NMF grouping of keywords (all italicized wording below is written by AI):

Government and Development Initiatives
   - Keywords: human, developing, partnership, model, needed, federal, government, standard, development, rd.

Financial and Operational Performance
   - Keywords: finding, model, capability, financial, operational, relationship, resource, innovation, firm, performance.

Labor and Employment Dynamics
   - Keywords: percent, warehousing, new, hiring, algorithm, labor, task, job, firm, worker.

Risk and Effectiveness Assessment
   - Keywords: assessment, capability, approach, effectiveness, financial, model, analysis, image, human, risk.

Research and Technological Applications

   - Keywords: researcher, computer, group, category, framework, article, machine, application, making, human.

Industry Challenges and Sustainability
  - Keywords: new, industry, initiative, challenge, sustainability, role, sustainable, model, development, change.


Results of Artificial Intelligence with Agent Assistance
Originally the intention was to run the code on an API format for ChatGpt 4omni, however the size of the articles and the amount of tokens sent have required this process to utilize Gpt3.5-turbo instead.  

The initial output of the AI agent team is a report on each article, and not quantitative as this is becoming a lengthy paper.  The file is saved as /content/meta_analysis_report.txt and here at the end of the paper as Supplemental 3. and then accessed in the next code block by another set of AI Agents that will process the data and create our final report that is contained in the file named /content/refined_meta_analysis_report.txt  or supplemental 4. 
Here are the themes that the AI Agent teams determined existed in our articles (all of the following content in italics is AI generated): Business Strategy
AI is fundamentally transforming business strategies across sectors such as finance, marketing, supply chain management, and e-commerce. It is revolutionizing decision-making processes, enhancing business agility, and fostering innovation.
National Strategies
Countries are developing strategic AI plans focusing on modernizing public sectors, responsible data management, and investing in AI capacity. These strategies highlight the proactive measures nations are taking to address AI's societal impacts.
Decision-Making 
AI is increasingly integrated into strategic decision-making, with tools like generative AI enhancing planning and stakeholder mobilization. However, the adoption rate varies, and there is a significant emphasis on the human-AI interaction.
Corporate Governance and Ethics 
The integration of AI into corporate governance emphasizes ethical decision-making, transparency, and accountability. There is a call for comprehensive frameworks to address these ethical concerns.
Financial Sector
AI's role in financial decision-making is growing, with a focus on explainable AI (XAI) to enhance transparency and mitigate biases.
Sustainability
AI is seen as a crucial tool for sustainable development, aligning with UN SDGs and addressing global challenges.
International Relations
AI is transforming international relations by automating foreign policy processes and enhancing predictive analytics. There is a need for international cooperation to address ethical and legal issues.
Regulatory and Policy Frameworks
There is a strong emphasis on the need for robust regulatory responses to manage AI's risks, including bias, data privacy, and cybersecurity. Proposals for international regulatory agencies like the International Artificial Intelligence Organization (IAIO) are discussed.

# Final results of all processes 
Compiling all of our data we have the following thematic groups from the articles:

# Interpretation 
Interpretation 1: the experiment 
In order to interpret these results, it is necessary to reorganize the above framework into like categories and see how many unique individual thematic elements there are.  

In relation to the experiment portion of this paper we can see some interesting results from the outcomes.  First from the chart above there are visible gaps between the types of research that were conducted.  The AI Agents output the largest amount of thematic elements, however three of the thematics that they put out all fell under “strategy” components; business, national and international.  If we combine that sector together we will have the following thematic groups: Human 6
LDA 6
NMF 6
AI Agents 6 It is worth noting that LDA and NMF have 6 elements each because that is what the element limit was determined to be based on running the coherence score, however we still had the same amount of thematics across all 4 methods of thematic research.  

Second, we can see that for the most part all methods of research yielded some similar thematic results, as well as results that were hidden from the manual thematic search.  Looking at it a bit more closely by grouping the different thematics together into more common alignment, we end up with nine clusters of thematics.  Here are the clustered themes grouped into unique thematic categories based on additional python based text preprocessing and KMeans clustering:
Cluster 1
- Financial and Operational Performance
Cluster 2
- Industry Challenges and Integration 
Cluster 3
- Strategy and Competition
- National and Public Sector Approaches
Cluster 4
-   Ethics
Corporate Governance and Ethics
Regulatory and Ethical Issues
Cluster 5
- Governance and Regulation
- Digital Innovation and Workforce
Cluster 6
- Human Resources and Culture
- Organizational Culture and Change
Cluster 7
- Safety, Security, and Risk Management
- Risk and Effectiveness Assessment
Cluster 8
- Research Frameworks and Human Factors
Cluster 9
Decision Making

These clusters represent how themes from different research methods align into broader thematic groups, indicating the overlap and unique contributions of each method. 

What is interesting is found in what is missing from the AI Agent framework; and that is the human resource aspect of the research.  This was a very prevalent theme in the direct reading and showed up again in the manual coding of the keywords, this theme was also there, albeit not directly named as “human resources” but, as “Labor and Employment Dynamics” in NMF and as “Research Frameworks and Human Factors” as well as “Organizational Culture and Change” in our LDA approach.  It is very hard to speculate on why this is, as there is very little understanding as to how the AI reasons and works out its logic; we could say that it doesn’t inherently care or think about humans just as easily as we could argue that the additional thematics it did pick up: “financial structures” and “sustainability” were in some way meant to incorporate human factors, perhaps it was using human resources as a method of expense and therefor financial structure.  Further research could easily figure this out, it may well be poor design or aspirational simple flaw in the code itself, as this is a novel approach to this type of AI Agent work and programming for myself.  

# Interpretation 2: The Thematics

Cluster 1: Financial and Operational Performance
This cluster focuses on how AI and human insights converge to enhance financial and operational performance within organizations. It highlights the integration of AI to optimize profitability, efficiency, and sustainability.  The following except from our articles provide a robust example of this thematic; “In recent times, the integration of AI, particularly Explainable Artificial Intelligence (XAI), has played a pivotal role in augmenting transparency and accountability within financial decision-making processes” (Rane, Choudhary, & Rane, 2023).

Cluster 2: Industry Challenges and Integration
Themes within this cluster seem to address the evolving challenges industries face as they integrate AI technologies. It may reflect on the adaptation required to align AI capabilities with industry norms and the strategic shifts necessary to harness AI's full potential in a competitive landscape. It is hard to say, these themes certainly appear as an overarching thematic element, but never seemed to appear directly as keywords or themes in any article.  This may be a reflection on reader bias. 

Cluster 3: Strategy and Competition
This cluster captures the dynamic interplay between strategic planning and competitive advantage in the AI context. AI and human researchers identified the strategic use of AI in maintaining competitive edges or producing them, as well as innovative practices, and foreseeing market trends. The following except is a sample of this thematic element from our articles; “Most plans detailed two aspects (a) how national governments should leverage AI to modernize the public sector, and (b) how industries and industrial sectors should take advantage of AI affordances to maintain and extend their competitiveness” (Fatima, D’Souza & Dawson, 2020).

Cluster 4: Ethics, Corporate Governance, and Regulatory Issues
Central to this cluster are the ethical considerations, governance structures, and regulatory frameworks essential for responsible AI deployment. It delves into the complexities of ethical AI use, the importance of robust corporate governance to oversee AI strategies, and the evolving regulatory landscape that guides AI applications. AI ethics are complex indeed and may have been the hardest part of create themes within this project.  The ethics of AI are not yet fully developed, extremely complex and vary between countries, so it was a hard element to categorize.  Here is an excerpt outlining this theme, “The co-existence of various institutional logics in AI strategies is a property of nascent AI systems which impacts the adoption of a regulatory approach and gives limited attention to developing an accountability framework”  (Radu, 2021).

Cluster 5: Governance and Regulation, Digital Innovation and Workforce
Themes in this cluster explore the intersection of governance, digital innovation, and workforce transformation. It highlights how AI influences governance practices, drives digital transformation, and necessitates new workforce strategies to accommodate the digital shift. Under the manual classifications I had all human aspects categorized under one central theme, human resources.  As discussed earlier there seems to be disagreements about how to handle the human resource for the machine side of categorizing. This area of thematic classification and the next are not themes I feel comfortable with the resultant clustering of. 

Cluster 6: Human Resources and Culture, Organizational Culture and Change
This cluster discusses the impact of AI on human resources and organizational culture. It emphasizes the transformative effect of AI on workplace dynamics, the integration of AI into cultural practices, and the need for cultural adaptation to maximize AI's organizational benefits. Here is an example of this from our articles, “Effectively implementing SHRM, which defines the structure of the plan by which the firm’s human resources intend to accomplish organisational goals, will ensure the organisation’s performance is sustainable” (Alnamrouti, Rjoub & Ozgit, 2022).

Cluster 7: Safety, Security, and Risk Management
Focusing on the critical aspects of safety, security, and risk management, this cluster examines how AI can be leveraged to enhance organizational safety protocols, mitigate risks, and ensure security in operations, underscoring the need for strategic risk assessment models in the age of AI. What this is not pointing out is that the theme does not set an internal or external locus of control as to the nature of the harm.  In other words, some articles regarding safety and risk management were about using AI to help maintain the security of an organization, while others were discussing the inherent dangers of AI itself. Here is an excerpt modeling the cluster as described in a positive manner towards AI, “The field of risk management is evolving, driven by technological advancements in AI and the ever-growing capabilities of models like Chatbots” (Yazdi, et al, 2024), and here is one modeling a negative view, “In hiring and credit, AI may amplify historical bias against female and minority background applicants, while in healthcare it may lead to opaque decisions because of its black box problem, to name just a few” (Firth-Butterfield, Matzo, 2020). 

Cluster 8: Research Frameworks and Human Factors
This cluster looks at the methodological and human factors in AI research and application. Possibly the frameworks used to study AI's impact, in relation to the human factors considered in AI integration, and the balance between human intuition and AI precision in research methodologies. I am unsure exactly, this may have come from hidden data. 

Cluster 9: Decision Making
This cluster explores how AI augments human decision-making capabilities within organizations. It details the role of AI in enhancing decision accuracy, speed, and effectiveness, transforming strategic decision processes with data-driven insights. This seemed to be a more prevalent area of thematics for manual research, but wasn’t rated as high with the other models. An example of this cluster (and prevalent keyword) can be seen in this except here “More than 40% of CEOs say they use generative AI to inform their decision-making processes" (Meissner & Narita, 2023).

# Conclusion 
We saw from our experiment here that there are benefits to using AI and machine learning when performing research.  Additionally, each of the clusters that we ended up with as the final interpretation of our lengthy research represents a facet of the overarching theme regarding the role of AI in organizational strategies and policy.  This has provided us with a comprehensive view of how artificial intelligence and human insight combine to collectively shape the academic and practical landscapes of business and technology.  

This paper could be used in a broader sense as an actual research submission if we were to address gaps in its method and weaknesses that were uncovered through the process. First there would need to be a more rigorous solution to bias in the manual approach to processing the papers, as well as a deeper look into the limitations of the LDA and NMF models used.  To try and mitigate the bias I ended up using a lot of extra stop words for the neural networks as I ran through iterations and for my personal reading, I was utilizing a very sloppy method of time to forget what I had previously done.  Further, a stronger synthesis of comparative analysis needs to be completed, as well as a deeper dive into quantitative structure of all data across the board and then of course this would lead to more interpretive and qualitative analysis.  


# References
Agrawal, A., Gans, J. S., & Goldfarb, A. (2023). Artificial intelligence adoption and system-wide 	
change. Journal of Economics & Management Strategy, 33 (1), 327–337. 
	doi.org/10.1111/jems.12521
Alnamrouti, A., Rjoub, H., & Ozgit, H. (2022). Do strategic human resources and artificial 
	intelligence help to make organisations more sustainable? Evidence from non-
	governmental organisations. Sustainability, 14(12), 7327. doi.org/10.3390/su14127327
Bellaouar, S., Bellaouar, M. M., & Ghada, I. E. (2021). Topic modeling: Comparison of LSA and 
	LDA on scientific publications. In Proceedings of the 2021 4th International Conference 
	on Data Storage and Data Engineering (pp. 59–64). Association for Computing 
	Machinery. doi.org/10.1145/3456146.3456156
Boukherouaa, E. B., AlAjmi, K., Deodoro, J., Farias, A., & Ravikumar, R. (2021). Powering the 
	Digital Economy: Opportunities and Risks of Artificial Intelligence in Finance. 
	Departmental Papers, 2021(024), A001. doi.org/10.5089/9781589063952.087.A001
Božić, V. (2023, April 18). AI and predictive analytics. Retrieved from: https://
	www.researchgate.net/publication/370074080_AI_and_Predictive_Analytics
Buxmann, P., Hess, T., & Thatcher, J. B. (2021). AI-based information systems. Business & 
	Information Systems Engineering, 63 (1), 1-4. doi.org/10.1007/S12599-020-00675-8
Chen, D., Esperança, J. P., & Wang, S. (2022). The impact of artificial intelligence on firm 
	performance: An application of the resource-based view to e-commerce firms. Frontiers 
	in Psychology, 13, 884830. doi.org/10.3389/fpsyg.2022.884830
Correia, A., & Água, P. (2023). Artificial intelligence to enhance corporate governance: A 
	conceptual framework. Corporate Board: Role, Duties and Composition, 19, 29-35. 
	doi.org/10.22495/cbv19i1art3
Gonesh, C., Saha, G., Menon, R., Paulin, S., Yerasuri, S. S., Saha, H., Dongol, P., & Lalit, S. 
	(2023). The impact of artificial intelligence on business strategy and decision-making 
	processes. Engineering, Economics and Logistics. doi.org/10.52783/eel.v13i3.386
Erdelyi, O., & Goldsmith, J. (2018, February 2-3). Regulating artificial intelligence: Proposal for 
	a global solution. Proceedings of the First AAAI/ACM Conference on Artificial 
	Intelligence, Ethics, and Society (pp. 95-101). doi.org/10.48550/arXiv.2005.11072
Fatima, S., Desouza, K. C., & Dawson, G. S. (2020). National strategic artificial intelligence 
	plans: A multi-dimensional analysis. Economic Analysis and Policy, 67, 178-194. 
	doi.org/10.1016/j.eap.2020.07.008
Firth-Butterfield, K., & Madzou, L. (2020, September 30). Rethinking risk and compliance for 
	the age of AI. World Economic Forum. Retrieved from: https://www.weforum.org/
	agenda/2020/09/rethinking-risk-management-and-compliance-age-of-ai-artificial-
	intelligence/
Gupta, P. (2024, January 5). The role of AI in crafting a modern data governance. International 
	Research Journal of Modernization in Engineering Technology and Science, 06, 
	2582-5208. Retrieved from: https://www.researchgate.net/publication/					377181716_THE_ROLE_OF_AI_IN_CRAFTING_A_MODERN_
	DATA_GOVERNANCE
Hlatshwayo, M. (2023, November 8). The integration of artificial intelligence (AI) into business 
	processes. Zenodo. Retrieved from: https://www.researchgate.net/publication/
	375489528_The_Integration_of_Artificial_Intelligence_AI_Into_Business_Processes
Kaggwa, S., Eleogu, T. F., Okonkwo, F., Farayola, O. A., Uwaoma, P. U., & Akinoso, A. 				
(2024). AI in Decision Making: Transforming Business Strategies. International Journal 			
of Research and Scientific Innovation (IJRSI), 10(12), 423. 
	doi.org10.51244IJRSI.2023.1012032
Kanitz, R., Gonzalez, K., Briker, R., & Straatmann, T. (2023). Augmenting organizational 
	change and strategy activities: Leveraging generative artificial intelligence. The Journal 
	of Applied Behavioral Science, 59 (3), 345–363. doi.org/10.1177/00218863231168974
Khan, A. (2023, November 13). The role of artificial intelligence in shaping modern business 
	strategies. Rushford Business School. Retrieved from https://rushford.ch/insights/the-
	role-of-artificial-intelligence-in-shaping-modern-business-strategies/
Kulkov, I., Kulkova, J., Rohrbeck, R., Menvielle, L., Kaartemo, V., & Makkonen, H. (2023). 
	Artificial intelligence-driven sustainable development: Examining organizational, 
	technical, and processing approaches to achieving global goals. Sustainable 
	Development. doi.org/10.1002/sd.2773
Makridakis, S. (2017). The forthcoming artificial intelligence (AI) revolution: Its impact on 
	society and firms. Futures, 90, 46-60. doi.org/10.1016/j.futures.2017.03.006
Meleouni, C., & Efthymiou, I. (2023). Artificial intelligence (AI) and its impact in international 
	relations. Journal of Politics and Ethics in New Technologies and AI, 2 (1), e35803. 
	doi.org/10.12681/jpentai.35803
Miller, S. (2021). Artificial intelligence in public services: State of the art and future research 			
directions. Policy & Society, 40(2), 178-194. doi.org/10.1093/polsoc/puab012
Norman, J. (n.d.). Newell, Simon & Shaw develop the first artificial intelligence program. 	
	History of Information. Retrieved from: 
	https://www.historyofinformation.com/detail.php?id=742
Oduro, S., De Nisco, A., & Mainolfi, G. (2023). Do digital technologies pay off? A meta-analytic 
	review of the digital technologies/firm performance nexus. Technovation, 128, 102836. 
	doi.org/10.1016/j.technovation.2023.102836
Office of Science and Technology Policy. (2023). National artificial intelligence research and 
	development strategic plan: 2023 update. The White House. Retrieved from; https://
	www.whitehouse.gov/wp-content/uploads/2023/05/National-Artificial-Intelligence-
	Research-and-Development-Strategic-Plan-2023-Update.pdf
Office of Science and Technology Policy. (2022). U.S.-EU Trade and Technology Council: 
	Executive Consultation on Artificial Intelligence. The White House. Retrieved from: 
	https://www.whitehouse.gov/wp-content/uploads/2022/12/TTC-EC-CEA-AI-
	Report-12052022-1.pdf
Olsen, D. R. (2006). A brief history of artificial intelligence. Retrieved from:
	 https://courses.cs.washington.edu/courses/csep590/06au/projects/history-ai.pdf
Öztürk, D. (2021). What does artificial intelligence mean for organizations? A systematic review 
	of organization studies research and a way forward. In S. Bozkuş Kahyaoğlu (Ed.), The 
	Impact of Artificial Intelligence on Governance, Economics and Finance, Volume I: 
	Accounting, Finance, Sustainability, Governance & Fraud: Theory and Application (pp.			
 265-289). Springer, Singapore. doi.org/10.1007/978-981-33-6811-8_14
Priya, B. C. (2022). Document-Term matrix in NLP: Count and TF-IDF scores explained. 
	Hackernoon. Retrieved from: https://hackernoon.com/document-term-matrix-in-nlp-
	count-and-tf-idf-scores-explained#
Radu, R. (2021). Steering the governance of artificial intelligence: National strategies in 
	perspective. Policy and Society, 40 (2), 178–193. 
	doi.org/10.1080/14494035.2021.1929728
Rane, N., Choudhary, S., & Rane, J. (2023, November 17). Explainable artificial intelligence 
	(XAI) approaches for transparency and accountability in financial decision-making. 	
	dx.doi.org/10.2139/ssrn.4640316
Shafiabady, N., Hadjinicolaou, N., Din, F. U., Bhandari, B., Wu, R. M. X., & Vakilian, J. (2023). 
	Using artificial intelligence (AI) to predict organizational agility. PLOS ONE, 18(5), 		
	e0283066. doi.org/10.1371/journal.pone.0283066
Shristikotaiah (2022). NLP Gensim tutorial - Complete guide for beginners. Geeks for 
	Geeks. Retrieved from; https://www.geeksforgeeks.org/nlp-gensim-tutorial-complete-
	guide-for-beginners/
Taurins, E. (2021). The impact of AI on strategic thinking and decision-		
	making. AIM Business School. Retrieved from; 
	https://www.elibrary.imf.org/view/journals/087/2021/024/article-A001-en.xml
Trunk, A., Birkel, H., & Hartmann, E. (2020). On the current state of combining human and 
	artificial intelligence for strategic organizational decision making. Business Research, 
	13, 875–919. doi.org/10.1007/s40685-020-00133-x
Walter, Y. (2024). Managing the race to the moon: Global policy and governance in artificial 			
intelligence regulation—A contemporary overview and an analysis of socioeconomic 			
consequences. Discover Artificial Intelligence, 4(14).  
	doi.org/10.1007/s44163-024-00109-4
Yang, Z., Hyman, M., Zhou, X., Li, G., & Munim, Z. (2022). Impact of artificial intelligence on 
	business strategy in emerging markets: A conceptual framework and future research 
	directions. International Journal of Emerging Markets, 17. 
	doi.org/10.1108/IJOEM-04-2022-995
Yazdi M, Zarei E, Adumene S, Beheshti A. Navigating the power of artificial intelligence in risk 
	management: A comparative analysis. Safety. 2024;10(2):42. 
	doi.org/10.3390/safety10020042
Zhao, J. (2024). Promoting more accountable AI in the boardroom through smart regulation. 	
	Computer Law & Security Review, 52, 105939. doi.org/10.1016/j.clsr.2024.105939


Supplement Section 

Supplemental 1
Data Collection & Search Strategy
Search Terms:
Core Terms: “artificial intelligence”, “AI”, ”machine learning”, "deep learning”, or “automation”
Organizational Focus: “global organizations”, “corporate strategies”, “organizational policy”, "strategic decision-making”, "policy development”, “business strategy”, “strategic policy” 
Impact Terms: “impact”, “effect”, “consequence”, “influence”
Qualitative Focus: “qualitative research”, “thematic analysis”
Additional Focus: “meta-analysis”
Exclusion Terms: Technical projects will be included initially to retain relevancy but may be excluded later if deemed necessary.
Databases:
- Google Scholar
- JSTOR
- ScienceDirect
- IEEE Xplore
- Web of Science
- Business Journals
- Professional Licensure Sites
- WEF (World Economic Forum)
- IMF (International Monetary Fund)
- US.GOV
Search Platforms:
- Google
- ChatGPT-4o
- Llama3 70b
Search Methodology:
Boolean Logic: Combine keywords with AND, OR, NOT to refine results.
   Example: “artificial intelligence” AND “global organizations” AND (impact OR influence)

Supplemental 2
Background of AI in Organizations 
Artificial Intelligence (AI) has experienced a significant evolution since its inception, deeply impacting organizational strategies and policies. Initially, in the 1950s and 1960s, AI began with foundational work in computer science and cognitive psychology. Alan Turing's seminal 1950 paper, "Computing Machinery and Intelligence," introduced the Turing Test, which sparked social interest in the potential of machine intelligence (Olsen, 2006). Early AI research focused on symbolic reasoning, leading to notable projects such as the Logic Theorist in 1955 and the General Problem Solver in 1957 (Norman, n.d.).

In the 1970s and 1980s, AI's application in organizational strategy advanced with the development of expert systems. These systems replicated human decision-making in specialized areas like medicine and finance. However, the field faced significant challenges during the 1980s and 1990s, a period known as the "AI Winter," marked by unmet expectations and funding cuts. Despite these setbacks, this era saw the initial development of neural networks and machine learning algorithms (Olsen, 2006).

The 2000s witnessed a boom in big data and advancements in computational power, which transformed AI's role in organizational strategy and policy (Olson, 2006). Machine learning and data analytics became integral to decision-making processes, particularly in finance and medicine. Organizations began to leverage predictive analytics and vast datasets, driving strategic adaptations based on data insights.

The integration of AI into business strategies accelerated in the 2010s, with technologies like natural language processing (NLP), recommendation systems, and autonomous agents becoming prevalent. AI's impact extended to areas such as data privacy and algorithmic bias, prompting organizations to develop ethical guidelines and compliance policies (Hlatshwayo, 2023). In recent years, AI has become a catalyst for strategic innovation, reshaping business models and competitive dynamics. Organizations now use AI for personalized customer experiences, operational efficiencies, and strategic foresight. Policy development has also emphasized governing AI's ethical and societal implications, further influencing organizational strategies.

In the current landscape, AI is leveraged for strategic decision-making, policy development, innovation, adaptation, workforce dynamics, and risk management. Studies have shown how organizations use AI for predictive analytics, scenario planning, and decision support systems, highlighting both benefits and challenges (Božić, 2023). Research has also focused on AI's role in data governance, ethical use, and regulatory compliance, illustrating how AI helps formulate and implement policies aligned with organizational goals (Gupta, 2024).

Studies have examined AI's effects on job redesign, skills development, and employee engagement, highlighting both positive and negative impacts on the workforce. Additionally, research has reviewed AI's use in identifying risks, mitigating threats, and responding to crises, demonstrating its effectiveness in various risk management scenarios.

Despite the extensive research, there are still gaps. For example, more studies are needed on AI's impact on strategic agility and its integration into decision-making processes. There is also a need for further research on balancing AI innovation with ethical considerations in policy-making, AI's impact on organizational culture during innovation and adaptation processes, and strategies for managing workforce transitions and developing employee skill rehabilitation programs. Furthermore, integrating AI into holistic risk management strategies and understanding AI's role in disaster recovery remain areas requiring additional exploration.


Supplemental 3

Refined_Meta_Analysis_Report.txt

### Meta-Analysis Report: Impact of Artificial Intelligence on Global Organizational Strategies and Policy

This report synthesizes the thematic elements, research gaps, and conclusions from various articles on the impact of Artificial Intelligence (AI) on global organizational strategies and policy.

#### Overarching Themes
1. **AI and Business Strategy**: AI is fundamentally transforming business strategies across sectors such as finance, marketing, supply chain management, and e-commerce. It is revolutionizing decision-making processes, enhancing business agility, and fostering innovation.
2. **National AI Strategies**: Countries are developing strategic AI plans focusing on modernizing public sectors, responsible data management, and investing in AI capacity. These strategies highlight the proactive measures nations are taking to address AI's societal impacts.
3. **AI in Decision-Making**: AI is increasingly integrated into strategic decision-making, with tools like generative AI enhancing planning and stakeholder mobilization. However, the adoption rate varies, and there is a significant emphasis on the human-AI interaction.
4. **Corporate Governance and Ethics**: The integration of AI into corporate governance emphasizes ethical decision-making, transparency, and accountability. There is a call for comprehensive frameworks to address these ethical concerns.
5. **AI in Financial Sector**: AI's role in financial decision-making is growing, with a focus on explainable AI (XAI) to enhance transparency and mitigate biases.
6. **AI and Sustainability**: AI is seen as a crucial tool for sustainable development, aligning with UN SDGs and addressing global challenges.
7. **International Relations and AI**: AI is transforming international relations by automating foreign policy processes and enhancing predictive analytics. There is a need for international cooperation to address ethical and legal issues.
8. **Regulatory and Policy Frameworks**: There is a strong emphasis on the need for robust regulatory responses to manage AI's risks, including bias, data privacy, and cybersecurity. Proposals for international regulatory agencies like the International Artificial Intelligence Organization (IAIO) are discussed.

#### Research Gaps
1. **Sector-Specific Impacts**: More research is needed to understand AI's specific impacts across different industries and sectors.
2. **Ethical and Legal Implications**: There is a lack of in-depth analysis on the ethical implications of AI adoption and the effectiveness of existing regulatory frameworks.
3. **AI in Emerging Markets**: Limited research on AI's challenges and opportunities in emerging markets and the interplay between institutional environments and AI-driven strategies.
4. **Human-AI Collaboration**: Further exploration is required on the dynamics of human-AI collaboration, especially in strategic decision-making and organizational change.
5. **Sustainability and AI**: Research should focus on balancing AI's benefits and risks in sustainability efforts, including human factors and global challenges.
6. **AI Governance**: The effectiveness of ethics-oriented AI governance compared to rule-based systems and the role of public-private interactions in shaping AI policies need further investigation.

#### Conclusions
1. **Transformative Potential**: AI is transforming business strategies, decision-making processes, and organizational performance across various sectors. Its integration requires aligning initiatives with strategic goals and addressing ethical considerations.
2. **Strategic AI Plans**: National AI strategies demonstrate a proactive approach to managing AI's societal impacts, emphasizing modernization, responsible data management, and capacity development.
3. **Ethical Governance**: Integrating AI into corporate governance can enhance ethical decision-making, but requires robust frameworks to address transparency and accountability issues.
4. **Financial Sector**: AI's role in financial decision-making is significant, with explainable AI techniques crucial for transparency and bias mitigation.
5. **Sustainability**: AI can significantly advance sustainable development, aligning with global goals and addressing pressing challenges.
6. **International Cooperation**: The need for international regulatory frameworks to manage AI’s ethical and legal issues is critical. Proposals like the IAIO aim to facilitate global cooperation.
7. **Balanced Approach**: A balanced approach that leverages AI's strengths while valuing human expertise is essential for effective risk management and organizational change.

This meta-analysis underscores the critical role of AI in reshaping global organizational strategies and policies, highlighting the need for continued research and robust regulatory frameworks to harness AI's potential responsibly.

Supplemental 2 
Meta_Analysis_Report.txt

This report provides a comprehensive synthesis of thematic elements from the articles reviewed by various experts in the field of organizational processes and AI.

Article 1 Summary:
The article discusses the impact of artificial intelligence (AI) on business strategy in emerging markets. AI technologies are revolutionizing various sectors like finance, marketing, and supply chain management. The article emphasizes the importance of understanding institutional environments and cultural forces in shaping AI-driven business strategies. It calls for more research to address the challenges and opportunities of AI in emerging markets.

Article 1 Thematic Elements:
Key thematic elements:
1. Impact of artificial intelligence (AI) on business strategy
2. Revolutionizing sectors such as finance, marketing, and supply chain management
3. Importance of understanding institutional environments and cultural forces in shaping AI-driven business strategies
4. Challenges and opportunities of AI in emerging markets

Research gaps:
1. Need for more research to address challenges and opportunities of AI in emerging markets

Methodologies used:
The article likely includes a literature review, case studies, and possibly qualitative analysis of AI applications in emerging markets.

Conclusions:
1. AI technologies are revolutionizing various sectors in emerging markets.
2. Understanding institutional environments and cultural forces is crucial for shaping AI-driven business strategies.
3. More research is needed to address the challenges

Article 2 Summary:
National strategic AI plans demonstrate how nations are preparing for the impact of AI on society. Plans focus on modernizing public sectors, managing data and algorithms responsibly, governance of AI systems, and investing in capacity development. These plans offer insights into how countries are proactively addressing challenges posed by rapid technological innovation.

Article 2 Thematic Elements:
Key thematic elements:
1. National strategic AI plans
2. Impact of AI on society
3. Modernizing public sectors
4. Responsible data and algorithm management
5. Governance of AI systems
6. Investing in capacity development
7. Proactive approach to addressing challenges of technological innovation

Research gaps:
1. Specific details on the strategies and measures outlined in the national AI plans
2. Long-term effectiveness and sustainability of the plans
3. Comparison of different countries' approaches and their outcomes

Methodologies used:
1. Content analysis of national strategic AI plans
2. Comparative analysis of different countries' approaches
3. Qualitative assessment of the strategies and initiatives outlined in the plans

Conclusions drawn:
1. National strategic AI

Article 3 Summary:
The article discusses the potential impact of the upcoming Artificial Intelligence (AI) Revolution, comparing it to past industrial and digital revolutions. It explores the predictions made in 1995, the advancement of AI technologies, and their potential to revolutionize society, firms, and employment. It also delves into the implications of AI on job displacement, economic growth, and societal structure, highlighting the need for proactive measures to mitigate risks and maximize benefits.

Article 3 Thematic Elements:
Key thematic elements:
1. The potential impact of the upcoming Artificial Intelligence (AI) Revolution
2. Comparison to past industrial and digital revolutions
3. Predictions made in 1995 and the advancement of AI technologies
4. Revolutionizing society, firms, and employment
5. Implications of AI on job displacement, economic growth, and societal structure
6. The need for proactive measures to mitigate risks and maximize benefits

Research gaps:
1. The article does not explicitly mention specific research gaps, but potential gaps could include:
   a. Lack of in-depth analysis on the specific impacts of AI across different industries
   b. Limited discussion on the ethical implications of AI adoption
   c. Insufficient exploration of potential policy interventions to

Article 4 Summary:
This article explores the relationship between artificial intelligence capability (AIC) and firm performance in e-commerce firms. It establishes a higher-order model of AIC and proposes a research model to study the impact of AIC on firm performance. Through data analysis using PLS-SEM, the study finds that AIC indirectly affects firm performance through firm creativity, artificial intelligence management (AIM), and AI-driven decision making (AIDDM). The study also highlights the moderating effects of innovation culture and environmental dynamism

Article 4 Thematic Elements:
Key thematic elements:
1. Relationship between artificial intelligence capability (AIC) and firm performance in e-commerce firms.
2. Higher-order model of AIC.
3. Impact of AIC on firm performance.
4. Indirect effects of AIC on firm performance through firm creativity, artificial intelligence management (AIM), and AI-driven decision making (AIDDM).
5. Moderating effects of innovation culture and environmental dynamism.

Research gaps:
1. The specific mechanisms through which AIC impacts firm performance.
2. The interplay between AIC, firm creativity, AIM, and AIDDM.
3. The influence of innovation culture and environmental dynamism on the relationship between AIC and firm performance.

Methodologies used:
1. Data analysis

Article 5 Summary:
The article discusses the importance of corporate governance (CG) and the potential of artificial intelligence (AI) to enhance ethical decision-making in corporations. It highlights the challenges and benefits of integrating AI into CG, proposing a comprehensive framework to address ethical and transparency issues. The research aims to empirically test the impact of AI on CG effectiveness.

Article 5 Thematic Elements:
Key thematic elements:
1. Corporate governance and its importance
2. Ethical decision-making in corporations
3. Integration of artificial intelligence (AI) in corporate governance
4. Ethical and transparency issues in corporate governance
5. Impact of AI on corporate governance effectiveness

Research gaps:
1. Lack of empirical evidence on the impact of AI on corporate governance effectiveness
2. Need for a comprehensive framework to address ethical and transparency issues in integrating AI into corporate governance

Methodologies used:
1. Empirical testing to assess the impact of AI on corporate governance effectiveness
2. Proposal of a comprehensive framework to address ethical and transparency issues

Conclusions drawn:
1. AI has the potential to enhance ethical decision-making in corporations
2. Challenges and

Article 6 Summary:
The article discusses the rise of Artificial Intelligence (AI) in financial decision-making and the importance of transparency and accountability. It explores Explainable AI (XAI) techniques to make AI systems more understandable and trustworthy in the financial sector. Various XAI methods, such as rule-based systems and SHAP values, are highlighted for their role in enhancing transparency and mitigating biases. The future of XAI in finance includes user-centric approaches, continuous monitoring, and education initiatives to ensure responsible AI deployment.

Article 6 Thematic Elements:
Key Thematic Elements:
1. Rise of Artificial Intelligence (AI) in financial decision-making
2. Importance of transparency and accountability in AI systems
3. Explainable AI (XAI) techniques in the financial sector
4. Various XAI methods for enhancing transparency and mitigating biases
5. Future of XAI in finance: user-centric approaches, continuous monitoring, and education initiatives

Research Gaps:
1. The article does not mention specific challenges or limitations of implementing XAI in financial decision-making.
2. It does not delve into the potential ethical considerations or regulatory implications associated with the use of AI in finance.

Methodologies Used:
1. The article discusses various XAI techniques such as rule-based systems and SHAP values.
2

Article 7 Summary:
The article highlights the critical role of artificial intelligence (AI) in decision-making for future competitiveness. Trust, access, and integration will shape AI adoption. Only 7% of companies currently use AI in big strategic decisions, but 40% of CEOs use generative AI. Human-AI interaction and determining which decisions to delegate to AI are key skills.

Article 7 Thematic Elements:
Key Thematic Elements:
1. Importance of artificial intelligence (AI) in decision-making for future competitiveness.
2. Trust, access, and integration as key factors shaping AI adoption.
3. Human-AI interaction and decision delegation as crucial skills in utilizing AI effectively.

Research Gaps:
1. Limited adoption of AI in big strategic decisions by companies.
2. Discrepancy between the low percentage of companies using AI in strategic decisions (7%) and the higher percentage of CEOs utilizing generative AI (40%).

Methodologies Used:
The article likely utilized a combination of qualitative and quantitative research methods to gather data on the current use of AI in decision-making processes. This could involve surveys, interviews, and data analysis to draw conclusions on AI adoption and

Article 8 Summary:
The article discusses the impact of AI on strategic thinking and decision-making in the corporate world. It highlights the rise of generative AI tools like ChatGPT and the potential for AI to automate strategic decision-making. The article emphasizes the need for strong strategic thinking skills in leaders and the importance of understanding the interconnection between humans and AI.

Article 8 Thematic Elements:
Key thematic elements:
1. Impact of AI on strategic thinking
2. Rise of generative AI tools
3. Automation of strategic decision-making
4. Importance of strong strategic thinking skills in leaders
5. Interconnection between humans and AI

Research gaps:
1. The article does not delve into specific case studies or real-world examples of AI impacting strategic decision-making in corporations.
2. It does not explore potential ethical implications or risks associated with relying heavily on AI for strategic thinking.

Methodologies used:
1. The article likely employs a literature review approach to discuss existing research and trends in AI and strategic decision-making.
2. It may also involve qualitative analysis of expert opinions or case studies to support its arguments.

Conclusions drawn:
1.

Article 9 Summary:
The article explores the transformative impact of Artificial Intelligence (AI) on strategic business decision-making, emphasizing how AI reshapes the corporate world. It discusses the emergence and evolution of AI in business strategy, highlighting its role in disrupting traditional decision models, enhancing business agility, and fostering inclusive practices. The study emphasizes the importance of developing an AI-ready corporate culture and sustainable business futures.

Article 9 Thematic Elements:
Key thematic elements: 
1. Impact of Artificial Intelligence (AI) on strategic business decision-making
2. Evolution of AI in business strategy
3. Disruption of traditional decision models
4. Enhancement of business agility
5. Inclusive practices in AI implementation
6. Development of an AI-ready corporate culture
7. Sustainable business futures

Research gaps:
1. Specific examples or case studies demonstrating the transformative impact of AI on decision-making processes
2. In-depth analysis of challenges and limitations in integrating AI into business strategy
3. Long-term effects of AI implementation on organizational structures and employee roles

Methodologies used:
1. Literature review on the evolution of AI in business strategy
2. Analysis of case studies or examples showcasing AI's

Article 10 Summary:
Artificial intelligence (AI) is transforming society and the economy, with applications in decision-making, service delivery, and opportunity evaluation. AI is predicted to have a significant impact on companies, with Forbes reporting 95% of executives foreseeing its importance. Ethical considerations, such as fairness and privacy, are crucial as AI becomes more integrated. Researchers are exploring AI's implications on organizations and society, focusing on AI readiness, virtual assistants, bias, and ethics. The future of AI research includes examining strategic

Article 10 Thematic Elements:
Key thematic elements:
1. Transformation of society and the economy by artificial intelligence (AI).
2. Importance of AI in decision-making, service delivery, and opportunity evaluation.
3. Ethical considerations such as fairness and privacy in AI integration.
4. Implications of AI on organizations and society including AI readiness, virtual assistants, bias, and ethics.

Research gaps:
1. The article does not specifically mention any research gaps, but potential areas for further exploration could include the long-term social impacts of AI, the effectiveness of AI in decision-making processes, and the development of ethical frameworks for AI implementation.

Methodologies used:
The article does not explicitly mention specific methodologies used in the research. However, typical methodologies in AI research may include data analysis, machine

Article 11 Summary:
Strategic organizational decision making in today's complex world involves a dynamic process characterized by uncertainty. Diverse groups of employees handle large amounts of information, with the potential support of artificial intelligence (AI). Research suggests AI can aid decision making, but challenges such as bias, ethical considerations, and the division of tasks between humans and machines must be addressed. Managers should focus on AI literacy, data transparency, and developing a clear strategy for AI integration, while also considering the changing roles and responsibilities of human decision makers

Article 11 Thematic Elements:
Key thematic elements:
1. Strategic organizational decision making in a complex, uncertain environment
2. Utilization of artificial intelligence (AI) in decision making
3. Challenges related to AI implementation, including bias and ethical considerations
4. The changing roles and responsibilities of human decision makers in relation to AI

Research gaps:
1. The impact of AI on decision-making processes and outcomes in different organizational contexts
2. Strategies for effectively addressing bias and ethical considerations in AI decision support systems
3. The optimal division of tasks between humans and machines in decision-making processes

Methodologies used:
1. Literature review to understand the current state of research on AI in decision making
2. Analysis of case studies or examples of AI implementation in organizations
3.

Article 12 Summary:
Artificial intelligence (AI) is transforming business strategies and decision-making by enhancing efficiency, accuracy, and innovation. The integration of AI requires aligning initiatives with strategic goals, addressing ethical considerations, and leveraging human-AI collaboration for superior outcomes. Organizations must invest in AI literacy and training to unlock the full potential of AI in strategic decision-making.

Article 12 Thematic Elements:
Key thematic elements:
1. Transformation of business strategies and decision-making through AI.
2. Importance of aligning AI initiatives with strategic goals.
3. Addressing ethical considerations in AI integration.
4. Leveraging human-AI collaboration for improved outcomes.
5. Necessity of investing in AI literacy and training for maximizing AI potential in decision-making.

Research gaps:
1. The article does not specifically address specific ethical considerations that need to be addressed in AI integration.
2. The article does not delve into the potential challenges or drawbacks of human-AI collaboration in decision-making.
3. There is a lack of discussion on the potential barriers organizations may face in investing in AI literacy and training.

Methodologies used:
The article does not mention specific methodologies used

Article 13 Summary:
This study conducted a systematic literature review on AI's role in sustainable development, aligning with the UN SDGs. The proposed conceptual model highlights organizational, technical, and processing aspects for integrating AI into sustainability efforts. Future research areas include human factors in AI adoption, addressing global challenges, case analyses, and balancing sustainability benefits and risks.

Article 13 Thematic Elements:
Key thematic elements:
1. AI's role in sustainable development
2. Integration of AI into sustainability efforts
3. Alignment with UN SDGs
4. Organizational, technical, and processing aspects of AI in sustainability
5. Balancing sustainability benefits and risks

Research gaps:
1. Human factors in AI adoption
2. Addressing global challenges
3. Case analyses in AI and sustainability

Methodologies used:
1. Systematic literature review

Conclusions drawn:
1. AI can play a significant role in advancing sustainable development aligned with UN SDGs.
2. Future research should focus on human factors, global challenges, case studies, and balancing benefits and risks of AI in sustainability efforts.

Article 14 Summary:
Artificial Intelligence (AI) is transforming International Relations by automating foreign policy processes, enhancing predictive analytics, and revolutionizing diplomatic activities like data analysis and conflict resolution. Nations like the USA and China are racing to gain an AI advantage for national security. However, ethical considerations and the need for international cooperation in AI development remain critical.

Article 14 Thematic Elements:
Key thematic elements:
1. Transformation of International Relations through AI technology
2. Automation of foreign policy processes
3. Enhancing predictive analytics
4. Revolutionizing diplomatic activities
5. National competition for AI advantage in national security
6. Ethical considerations in AI development
7. Need for international cooperation in AI development

Research gaps:
1. The article does not specify specific examples of how AI is being utilized in foreign policy processes, predictive analytics, and diplomatic activities.
2. It does not delve into the potential risks and challenges associated with the increased reliance on AI in International Relations.
3. The article does not explore the potential consequences of a lack of international cooperation in AI development.

Methodologies used:
The summary does not provide information on

Article 15 Summary:
Digital technologies (DTs) positively impact firm performance, with a greater effect on innovation than operational efficiency or financial performance. AI has the highest impact, followed by BDAs, IoTs/CPS, and 3DP. DTs' performance improves over time, influenced by firm size, sector,

Article 15 Thematic Elements:
Key thematic elements:
1. Impact of digital technologies on firm performance
2. Influence of different digital technologies on innovation, operational efficiency, and financial performance
3. Factors affecting the performance of digital technologies over time

Research gaps:
1. The specific mechanisms through which different digital technologies impact firm performance
2. The interaction between firm characteristics (size, sector) and the performance of digital technologies
3. Potential differences in the impact of digital technologies across different industries or sectors

Methodologies used:
1. Analysis of the impact of different digital technologies on firm performance
2. Evaluation of the performance of digital technologies over time
3. Examination of the influence of firm size and sector on the performance of digital technologies

Conclusions:
1. Digital technologies

Article 16 Summary:
The article discusses the potential of generative artificial intelligence (GAI) in shaping organizational change work. It uses a case example of a culture change initiative to illustrate how GAI tools can enhance planning, stakeholder mobilization, and progress monitoring. The article highlights limitations of GAI and suggests future research directions on stakeholder responses, GAI impact on change work, and value creation.

Article 16 Thematic Elements:
Key thematic elements:
1. Generative artificial intelligence (GAI) in organizational change work
2. Use of GAI tools for planning, stakeholder mobilization, and progress monitoring
3. Limitations of GAI in organizational change work
4. Future research directions on stakeholder responses, GAI impact on change work, and value creation

Research gaps:
1. Limited understanding of stakeholder responses to GAI in organizational change work
2. Unclear impact of GAI on change initiatives
3. Lack of knowledge on how GAI contributes to value creation in organizational change efforts

Methodologies used:
1. Case example of a culture change initiative to illustrate the use of GAI tools
2. Analysis of how GAI can

Article 17 Summary:
US and EC at US-EU Trade and Technology Council in Sept 2021 plan joint study on AI impact on workforces. Pittsburgh statement commits to economic study on AI effects on employment and wages.

Article 17 Thematic Elements:
Key thematic elements:
1. AI impact on workforces
2. Economic study on AI effects on employment and wages

Research gaps:
1. The specific focus and scope of the joint study on AI impact on workforces are not provided in the summary.

Methodologies used:
1. The summary does not mention specific methodologies used in the joint study or economic study.

Conclusions drawn:
1. The US and EC have planned a joint study on the impact of AI on workforces, with a focus on employment and wages.
2. The Pittsburgh statement commits to conducting an economic study on the effects of AI on employment and wages.

Article 18 Summary:
The article discusses the evolving role of AI in modern business strategies, its impact on the global economy, and the ongoing debate over whether AI will replace human workers. It emphasizes the importance of AI-human collaboration, ethical considerations, and the need for businesses to leverage AI for efficiency and strategic decision-making.

Article 18 Thematic Elements:
Key thematic elements:
1. The evolving role of AI in modern business strategies
2. Impact of AI on the global economy
3. AI-human collaboration
4. Ethical considerations in AI adoption
5. Leveraging AI for efficiency and strategic decision-making

Research gaps:
1. The article does not delve into specific industries or sectors where AI is making the most significant impact.
2. It does not address potential policy implications or regulations related to AI adoption.

Methodologies used:
1. The article likely includes a review of existing literature and industry trends related to AI in business strategies.
2. It may also incorporate case studies or examples of successful AI implementation in businesses.

Conclusions drawn:
1. AI is becoming increasingly integrated into modern business strategies

Article 19 Summary:
The study examines the impact of organisational learning (OL) and corporate social responsibility (CSR) on non-governmental organisations' (NGOs) sustainability performance, with a focus on strategic human resource management (SHRM) practices and artificial intelligence (AI). Findings show a direct positive relationship between OL, CSR, SHRM, AI, and sustainable organisational performance. The study provides practical guidance for NGOs to enhance sustainability.

Article 19 Thematic Elements:
Key thematic elements:
1. Organisational learning (OL)
2. Corporate social responsibility (CSR)
3. Strategic human resource management (SHRM) practices
4. Artificial intelligence (AI)
5. Non-governmental organisations' (NGOs) sustainability performance

Research gaps:
1. The specific mechanisms through which OL, CSR, SHRM practices, and AI impact sustainability performance in NGOs may not have been fully explored.
2. The potential challenges or limitations of implementing these practices in NGOs may not have been thoroughly investigated.

Methodologies used:
1. Quantitative analysis to examine the relationships between OL, CSR, SHRM practices, AI, and sustainability performance.
2. Likely use of surveys, data analysis, and statistical techniques to gather and

Article 20 Summary:
AI is a powerful technology with risks. The federal government must invest in R&D to promote responsible innovation and address societal challenges that other sectors won't tackle.

Article 20 Thematic Elements:
Key thematic elements:
1. Artificial Intelligence (AI) as a powerful technology with associated risks.
2. The need for federal government investment in Research and Development (R&D) to promote responsible innovation.
3. Addressing societal challenges that other sectors may not be addressing.

Research gaps:
1. The specific areas within AI that pose risks and require responsible innovation.
2. Identification of societal challenges that are not being adequately tackled by other sectors.

Methodologies used:
1. The article does not specify any specific methodologies used, but likely involves a review of existing literature and policy analysis.

Conclusions drawn:
1. Emphasizing the importance of federal government investment in R&D for AI to ensure responsible innovation.
2. Highlighting the need to address societal

Article 21 Summary:
This article explores the benefits of accountable AI in corporate boardrooms and the need for regulation to promote accountability. It discusses the interconnections between AI, board decisions, legal risks, and regulatory frameworks. The focus is on achieving sustainable AI development through monitoring and mitigating risks in a corporate setting.

Article 21 Thematic Elements:
Key thematic elements:
1. Benefits of accountable AI in corporate boardrooms
2. The importance of regulation to promote accountability
3. Interconnections between AI, board decisions, legal risks, and regulatory frameworks
4. Sustainable AI development through risk monitoring and mitigation in a corporate setting

Research gaps:
1. The specific challenges in implementing and enforcing AI accountability in corporate governance
2. The effectiveness of existing regulatory frameworks in ensuring AI accountability in boardrooms

Methodologies used:
1. Literature review on AI accountability and corporate governance
2. Analysis of legal and regulatory frameworks related to AI governance
3. Case studies or examples illustrating the impact of AI on board decisions and legal risks

Conclusions drawn:
1. There is a need for regulation to

Article 22 Summary:
The article discusses the need for an international regulatory agency to address legal and ethical issues surrounding artificial intelligence (AI). It proposes the establishment of the International Artificial Intelligence Organization (IAIO) to coordinate global AI regulation efforts. The IAIO should start with soft law instruments and minimalistic administrative functions to facilitate international cooperation on AI policies.

Article 22 Thematic Elements:
Key thematic elements:
1. Need for international regulatory agency for AI
2. Legal and ethical issues surrounding AI
3. Establishment of the International Artificial Intelligence Organization (IAIO)
4. Coordination of global AI regulation efforts
5. Soft law instruments and minimalistic administrative functions

Research gaps:
1. How the IAIO would be structured and governed
2. Specific legal and ethical issues that the IAIO would address
3. Potential challenges in implementing global AI policies
4. Comparative analysis with existing international regulatory agencies

Methodologies used:
1. Proposal for the establishment of the IAIO
2. Discussion of soft law instruments and minimalistic administrative functions
3. Analysis of the need for international cooperation on AI policies

Conclusions drawn

Article 23 Summary:
The article explores national strategies on artificial intelligence (AI) from a hybrid governance perspective, analyzing approaches of countries like Canada and China. It highlights a shift towards ethics-oriented rather than rule-based systems in AI governance, with a focus on public-private interaction and the creation of new oversight institutions. The study emphasizes the complexity of AI governance, the dominance of industry in strategy development, and the need for clearer regulatory frameworks and accountability mechanisms.

Article 23 Thematic Elements:
Key thematic elements:
1. National strategies on artificial intelligence (AI)
2. Hybrid governance perspective
3. Shift towards ethics-oriented AI governance
4. Public-private interaction
5. Creation of new oversight institutions
6. Complexity of AI governance
7. Dominance of industry in strategy development
8. Clearer regulatory frameworks and accountability mechanisms

Research gaps:
1. The effectiveness of ethics-oriented AI governance compared to rule-based systems
2. Impact of public-private interaction on AI governance outcomes
3. Role and influence of new oversight institutions in AI governance
4. Evaluation of regulatory frameworks and accountability mechanisms in AI governance

Methodologies used:
1. Comparative analysis of national strategies on AI from countries like Canada and China
2. Examination of

Article 24 Summary:
The study explores the impact of AI in risk management, focusing on AI's role in image data analysis. While AI, particularly AIc-4, offers rapid and relevant risk assessments, it falls short in accuracy, practicality, and contextual understanding compared to human experts. Challenges such as accountability, bias, and interpretability must be addressed for effective integration of AI in risk management. The study advocates for a balanced approach that leverages AI's strengths while valuing human expertise, emphasizing the need for ongoing

Article 24 Thematic Elements:
Key thematic elements:
1. Impact of AI in risk management
2. AI's role in image data analysis
3. Comparison of AI (specifically AIc-4) and human experts in risk assessment
4. Challenges in AI integration in risk management (accuracy, practicality, contextual understanding, accountability, bias, interpretability)
5. Advocacy for a balanced approach combining AI strengths with human expertise

Research gaps:
1. Need for addressing challenges such as accountability, bias, and interpretability in AI for effective risk management
2. Lack of contextual understanding and accuracy in AI risk assessments compared to human experts

Methodologies used:
1. Comparative analysis of AI (AIc-4) and human experts in risk assessment
2.

Article 25 Summary:
The article explores the rapid adoption of artificial intelligence (AI) and machine learning (ML) in the financial sector, highlighting benefits like efficiency and deepening financial services, but also concerns like widening the digital divide. It discusses risks such as bias, explainability, cybersecurity, data privacy, and the potential impact on financial stability, emphasizing the need for robust regulatory responses and collaboration among stakeholders to address these challenges.

Article 25 Thematic Elements:
Key thematic elements:
1. Rapid adoption of artificial intelligence and machine learning in the financial sector
2. Benefits such as efficiency and deepening financial services
3. Concerns including widening the digital divide, bias, explainability, cybersecurity, and data privacy
4. Potential impact on financial stability
5. Emphasis on the need for robust regulatory responses and collaboration among stakeholders

Research gaps:
1. The article does not specifically mention any gaps in current research, but potential gaps could include further exploration of the specific effects of AI and ML in different areas of the financial sector, as well as the effectiveness of current regulatory responses in addressing emerging challenges.

Methodologies used:
1. The article likely employs a qualitative research approach, drawing on existing literature,

Article 26 Summary:
AI is transforming risk management and compliance, yet poses new risks like bias amplification and opaque decisions. Integrated audit software is crucial for managing these risks. Maximizing AI benefits while mitigating risks requires appropriate audit software documenting AI behavior, assessing compliance, and enabling cross-department collaboration. Responsible AI still requires human judgment.

Article 26 Thematic Elements:
Key thematic elements:
1. AI's transformation of risk management and compliance
2. Risks associated with AI, such as bias amplification and opaque decisions
3. Importance of integrated audit software in managing AI-related risks
4. Maximizing AI benefits while mitigating risks
5. Role of responsible AI and human judgment

Research gaps:
1. The article does not specifically mention any research gaps.

Methodologies used:
1. The article does not mention specific methodologies used, but it discusses the importance of integrated audit software for managing AI-related risks.

Conclusions drawn:
1. AI is bringing significant changes to risk management and compliance processes.
2. AI introduces new risks, including bias amplification and opaque decisions.
3. Integrated audit software is


