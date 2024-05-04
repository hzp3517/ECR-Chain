# CEE_supp 4-shot

在RECCON的train set中选4个例子作为examples，dialog id分别是：tr_1287_9_cause6, tr_3676_5_cause3, tr_20_8_cause8, tr_223_2_cause1

下面的示例prompt中用的测试数据，来自于test set中dialog id为 tr_2347_2_cause2 的sample。

4步推理流程：exlanation -> stimulus -> appraisal -> reaction

<br>

---
示例prompt：

````
Please understand the emotion cause for the target utterance in a given conversation.

In a given conversation, each sentence contains its index number, speaker, emotion, and its content, written in this format: `#[index number]: [speaker] ([emotion]): "[content]"`. A target non-neutral utterance will be specified, where the emotion of the target non-neutral utterance may be one of the types of [happiness, sadness, anger, fear, surprise, disgust]. The target speaker is the corresponding speaker of the target utterance. Besides, a causal utterance will be given. A causal utterance is a particular utterance in the conversation history (can be the target utterance itself) responsible for the non-neutral emotion in the target utterance. You should find what the exact cause factor is from the causal utterance and understand how the factor causes the target speaker's emotional response reflected in the target utterance.

Specifically, you should perform a 4-step reasoning process:
1. Give an "Explanation": First, understand the semantics expressed by the cause utterance based on the conversation context. Then, consider what exact factor conveyed by the cause utterance might be responsible for the non-neutral emotion in the target utterance. You need to consider the factor in conjunction with the target speaker's inner thoughts and his/her reaction or behavior corresponding to the emotion in the target utterance. But note that if the speaker of the causal utterance is also the target speaker, the exact factor is more likely to be some objective events he/she described or his/her own subjective opinions, rather than his/her own actions. Because generally speaking, one's actions would not cause his/her own subsequent emotions, only his/her opinions or other events (including the other speaker's actions) can do. Use a few sentences to give your "Explanation" of the target speaker's emotion.
2. Extract the "Stimulus" from the causal utterance: A "Stimulus" can be an event, situation, opinion, or experience in the conversational context that is primarily responsible for the elicited emotion in the target utterance. Apart from them, the "Stimulus" could also be the other speaker’s counterpart reaction towards an event cared for by the target speaker. Write the "Stimulus" in one brief sentence. Note that it should correspond to the exact factors you found in the "Explanation".
3. Infer the "Appraisal" of the target speaker in the target utterance: Infer and describe the inner thoughts of the target speaker when the target utterance is saying based on the conversation context, the "Explanation" and the "Stimulus" you have written before. Write the "Appraisal" in one brief sentence.
4. Describe the "Reaction" of the target speaker in the target utterance: Describe the target speaker's reaction or behavior with his/her target emotion when the target utterance is saying according to the context, the "Explanation" and the "Appraisal" you have written before. Write the "Reaction" in one brief sentence.

I will show you some examples:

--- Example1 ---
Conversation:
"""
#1: A (neutral): "How's your business ?"
#2: B (disgust): "Everything seems to be getting worse . I don't know what to do with it ."
#3: A (neutral): "Things are tough all over . You should never give up !"
#4: B (neutral): "But how can I prevent a deficit ?"
#5: A (happiness): "I think you should make a thorough market investigation now ."
#6: B (happiness): "OK , OK . I ' ll do my best ."
#7: A (happiness): "I always consider you are such a guy who never gives up ."
#8: B (happiness): "I know life is not all roses , and I believe I will pull this through ."
#9: A (happiness): "Yeah , you can make it ."
"""
Target Utterance:
"""
#9: A (happiness): "Yeah , you can make it ."
"""
Causal Utterance:
"""
#6: B (happiness): "OK , OK . I ' ll do my best ."
"""
Now please perform the 4-step reasoning process:
Explanation:
The two speakers are talking about B's business. A advised B to make a market investigation to prevent a deficit. In the cause utterance #6, B accepted A's suggestion and promised to do his best. B's performance of not being afraid of difficulties moved A and made A willing to believe that he could succeed. Therefore, in the target utterance #9, A happily encourages B that he can make it.
Stimulus in #6:
- B promises to do his best.
Appraisal of B in #9:
- A thinks B is not afraid of difficulties and will succeed in the end.
Reaction of B in #9:
- A expresses his happiness and encouragement to B.

--- Example2 ---
Conversation:
"""
#1: A (neutral): "Oh , God . It ' s late . I ' m afraid I have to leave ."
#2: B (neutral): "But you just got here . Can ' t you stay a little longer and have some tea ?"
#3: A (happiness): "That ' s very kind of you . But if I don ' t go now I ' ll miss the last bus ."
#4: B (neutral): "That ' s too bad ."
#5: A (happiness): "Thank you for your hospitality ."
"""
Target Utterance:
"""
#5: A (happiness): "Thank you for your hospitality ."
"""
Causal Utterance:
"""
#3: A (happiness): "That ' s very kind of you . But if I don ' t go now I ' ll miss the last bus ."
"""
Now please perform the 4-step reasoning process:
Explanation:
In this conversation, B is tring to persuade A to stay a while. In the cause utterance #3, A was pleased by B's effort and think that B is very kind. It's A's view to B that made A want to thank B. So in the target utterance #5, A expresses his happiness and gratitude for B's hospitality.
Stimulus in #3:
- A's view that B is very kind.
Appraisal of A in #5:
- A thinks B is very kind.
Reaction of A in #5:
- A expresses his happiness and gratitude for B's hospitality.

--- Example3 ---
Conversation:
"""
#1: A (neutral): "Here ' s your hot dog and beer . What happened ? Did I miss anything ?"
#2: B (neutral): "Yeah , Cal Ripen just hit a home run ."
#3: A (neutral): "What ' s the score ?"
#4: B (neutral): "Well it was 3 to 4 , but Ripen ' s home run made it 5 to 4 since another player was on first base ."
#5: A (neutral): "So Baltimore is winning ?"
#6: B (happiness): "Right ."
#7: A (happiness): "This is a really great place to watch a baseball game ."
#8: B (happiness): "Yeah , there isn ' t a bad seat in the place ."
"""
Target Utterance:
"""
#8: B (happiness): "Yeah , there isn ' t a bad seat in the place ."
"""
Causal Utterance:
"""
#8: B (happiness): "Yeah , there isn ' t a bad seat in the place ."
"""
Now please perform the 4-step reasoning process:
Explanation:
The two speakers are chatting while watching a baseball game. A said that this is a good place to watch a baseball game. In the cause utterance #8, B agrees with A's opinion because he thinks that there isn't a bad seat in the place. It's B's view of the watching place that makes B feel happy and praise for this watching place in the target utterance #8.
Stimulus in #8:
- B's view that there isn't a bad seat in the place they sit in.
Appraisal of B in #8:
- B thinks that the place they sit in is good.
Reaction of B in #8:
- B happily praise for their watching place.

--- Example4 ---
Conversation:
"""
#1: A (neutral): "Jane , I was wondering if you you had any plans for saturday afternoon ."
#2: B (neutral): "A friend and I are planing to go out , why ? what's up ?"
#3: A (neutral): "There is a special exhibition of french sculptures at the museum , I was hoping you ' d like to come with me ."
#4: B (sadness): "I am afraid I can't I am going to be out all day ."
"""
Target Utterance:
"""
#4: B (sadness): "I am afraid I can't I am going to be out all day ."
"""
Causal Utterance:
"""
#3: A (neutral): "There is a special exhibition of french sculptures at the museum , I was hoping you ' d like to come with me ."
"""
Now please perform the 4-step reasoning process:
Explanation:
The two speakers are talking about plans for Saturday afternoon. In the cause utterance #3, A told B that there was a special exhibition and invited B to go with him. However, we know that B will have other arrangements on that day according to the target utterance #4. So A's invitation may make B feel regretful, which leads B to express his disappointments and regrets for not going with A in the target utterance #4.
Stimulus in #3:
- A invites B to go to the exhibition together.
Appraisal of B in #4:
- B thinks it is regretful to decline A's invitation.
Reaction of B in #4:
- B regretfully explains the reason why he cannot go with A.

--- To be solved ---
Conversation:
"""
#1: A (happiness): "The band is pretty good . How do you like the music ?"
#2: B (happiness): "It's very nice . I haven't heard live music in a while . This is fun ."
"""
Target Utterance:
"""
#2: B (happiness): "It's very nice . I haven't heard live music in a while . This is fun ."
"""
Causal Utterance:
"""
#1: A (happiness): "The band is pretty good . How do you like the music ?"
"""
Now please perform the 4-step reasoning process:

````