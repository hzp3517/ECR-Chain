# CEE_simplify_1shot

以CausalLM的方式，用生成式模型，以backward CoT的方式做Causal Emotion Entailment任务。

backward CoT：5步推理过程。theme -> reaction -> appraisal -> stimuli -> cause utt ids

在RECCON的train set中选1个例子作为example，dialog id是：tr_264_6
下面的示例prompt中用的测试数据，来自于test set中dialog id为 tr_9708_4 的sample。

注意：和CEE_backward_1shot的prompt的区别：这里在评价项和刺激项的描述中不加入要对应上上一项某项的要求

<br>

---
示例prompt：

````
Please understand the emotion cause for the target utterance in a given conversation.

In a given conversation, each utterance contains its index number, speaker, emotion, and its content, written in this format: `#[index number]: [speaker] ([emotion]): ["content"]`. Besides, a target non-neutral utterance will be specified, where the emotion of the target non-neutral utterance may be one of the types of [happiness, sadness, anger, fear, surprise, disgust]. You should predict which particular utterances in the conversation history (including the target itself) are causal utterances that are responsible for the non-neutral emotion in the target utterance.

Specifically, you should perform a 5-step reasoning process:
1. Output the "Theme": Summarize the conversation theme with a few words.
2. List "Reactions" items of target utterance: Describe the target speaker's reaction or behavior with his/her target emotion when the target utterance is saying according to the conversation theme and the context. List them by items.
3. List "Appraisals" items of target utterance: Infer and describe the inner thoughts of the target speaker when the target utterance is saying based on the context and the "Reactions" items listed above. List them by items.
4. List "Stimuli" items with their corresponding utterance id: The "Stimuli" items can be events, situations, opinions, or experiences in the conversational context that are primarily responsible for the "Appraisals" items listed above. Apart from them, the "Stimuli" items could also be the other speaker’s counterpart reactions toward events that are cared for by the target speaker and can cause these "Appraisals" items of the target speaker. List them by items.
5. Based on these "Stimuli" items, output the index number of the causal utterances in the form of Python list without any other content.

I will show you an example:

--- Example ---
Conversation:
"""
#1: A (neutral): "Hey , Mary , can you cut that out ?"
#2: B (neutral): "Cut what out I ' m not doing anything ."
#3: A (anger): "The tapping of your pen on your desk . It ' s driving me crazy ."
#4: B (anger): "Fine ! By the way would you mind not slurping your coffee every time you have a cup !"
#5: A (anger): "I don ' t slurp my coffee . And plus , how can you hear it when you ' re shouting into your phone all the time ?"
#6: B (anger): "You ' Ve got to be kidding me ! You ' re complaining about me talking on the phone when you go out for a cigarette break ten times a day to shoot the breeze ?"
"""
Target Utterance:
"""
#6: B (anger): "You ' Ve got to be kidding me ! You ' re complaining about me talking on the phone when you go out for a cigarette break ten times a day to shoot the breeze ?"
"""
Now please perform the 5-step reasoning process:
Theme: 
Complain about each other's noise.
Reactions of B in #6:
1. B expresses his anger with A's complaint to B.
2. B angrily points out that A chats frequently which also makes much noise.
Appraisals of B in #6:
1. B thinks the noise made by A is annoying.
2. B thinks that A has no right to blame B for making noise as A is also used to making noise.
Stimuli:
1. A is used to slurping his coffee which makes some annoying noise (#4).
2. A accused B of tapping the pen on the desk (#3).
3. A accused B of making noise during the phone call (#5, #6).
Causal utterances:
[3, 4, 5, 6]

--- To be solved ---
Conversation:
"""
#1: A (neutral): "The blake's got divorced ."
#2: B (neutral): "Really ? Why ?"
#3: A (neutral): "Mr . black has been getting a little around aside ."
#4: B (surprise): "I'm surprised . He does't look like a guy who'd ever cheat on his wife , does he ?"
"""
Target Utterance:
"""
#4: B (surprise): "I'm surprised . He does't look like a guy who'd ever cheat on his wife , does he ?"
"""
Now please perform the 5-step reasoning process:
````

<br>

---
示例response：
````
Theme:
Divorce and infidelity.
Reactions of B in #4:
1. B expresses surprise about Mr. Black's infidelity.
2. xxx.
Appraisals of B in #4:
1. B thinks that Mr. Black doesn't seem like the type of person who would cheat on his wife.
2. xxx.
Stimuli:
1. A informs B about Mr. Black's infidelity (#3).
2. xxx (#4).
Causal utterances:
[3, 4]
````
