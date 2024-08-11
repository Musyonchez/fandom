import spacy
from transformers import pipeline

print(spacy.util.get_installed_models())

# Load NLP model
nlp = spacy.load('en_core_web_sm')

# Define function for Named Entity Recognition
def extract_entities(chapter_text):
    doc = nlp(chapter_text)
    entities = {}
    for ent in doc.ents:
        if ent.label_ in ["PERSON", "ORG", "GPE"]:
            if ent.label_ not in entities:
                entities[ent.label_] = []
            entities[ent.label_].append(ent.text)
    return entities

# Define function for summarizing character and plot points
def summarize_chapter(chapter_text):
    summarizer = pipeline("summarization")
    summary = summarizer(chapter_text, max_length=200, min_length=50, do_sample=False)
    return summary[0]['summary_text']

# Example usage
chapter_text = """
 Three years later, Liu Yu's appearance hasn't changed much, he still looks like a teenager, but his figure is much stronger than before.

The    cultivation base has also reached the peak of the sixth level of qi refining, and is only one step away from the latter stage of qi refining.

   In fact, with the support of a large number of medicinal pills, he cultivated to the peak of the sixth level of Qi refining as early as eight months ago, but he encountered a bottleneck, and he could not make further progress after that.

   had to polish the mana carefully to make it more pure. During the period, it attacked the later stage several times, but all of them failed.

   With Liu Yumu's aptitude of fire, earth and three spiritual roots, it is not surprising that he encountered a bottleneck in the later stage of qi refining.

   "It is said that Tianlinggen has been cultivating until the Nascent Soul stage, and there will be no bottlenecks. It is indeed a favored son of heaven!"

In the    cave mansion, Liu Yu once again failed to hit the bottleneck, thinking a little depressed.

   The aptitude of the three spiritual roots, he wanted to naturally break through the bottleneck through cultivation, but he didn't know how much time it would take. He decided that he couldn't go on like this.

  According to the information Liu Yu had read in the Tibetan Sutra Pavilion, there are three ways to break through the bottleneck at the moment:

  The first is to break through through the battle between life and death.

   There is a great terror between life and death, which can greatly stimulate the potential of monks, and the probability of breaking through the bottleneck will increase a lot. However, not everyone can face the great terror between life and death calmly, so there are still a few people who choose this method.

   The second is to buy medicinal pills that break through the bottleneck, and use the power of medicinal pills to dissolve the bottleneck and achieve a breakthrough.

   However, the medicinal pills that broke through the bottleneck have always been in short supply and rarely appeared. Once they appeared, they would cause many cultivators to compete. The price was often extremely high. This spiritual stone was not something that ordinary cultivators could afford.

  The third is to carefully polish the mana and repeatedly hit the bottleneck.

   Dripping water to pierce the stone, so it is possible to succeed over time, but it takes a long time or a short time, and it is full of uncertainty.

   This is also the method commonly used by most monks. After all, the pills to break through the bottleneck are expensive, and at least five hundred spirit stones are used as a base, leaving countless ordinary monks helpless.

   Liu Yu was inclined towards the second type. He decided to go to the nearby market to see if he could buy an elixir that could break through the bottleneck.

   At the same time, he didn’t give up on polishing his mana. Maybe the bottleneck would be loosened that day?

   If these two methods fail, then for the sake of the great road, Liu Yu will participate in some dangerous tasks and break through the bottleneck through dangerous fighting methods!

  Wealth and wealth are sought in danger, and Taoism is taken in danger!

  …

   In the past three years, Liu Yu has already entered the beginning of the practice of Invisibility, reaching the first level.

   A monk who can hide himself from a small level, but must be in the same big realm,

   He is in the middle stage of qi refining now, even a monk in the late stage of qi refining cannot see through his cultivation, unless he is a cultivator in the perfect state of qi refining.

   Liu Yu used the hidden spirit technique to hide the fluctuations of mana and spiritual pressure.

  I scanned it with my divine sense, but it was difficult to find the hidden mana aura, so I felt relieved.

   With his current consciousness that is slightly better than the late stage of Qi refining, he cannot see through the truth, so ordinary Qi refining cultivators should not be able to find it.

   At the same time, he also yearns for the ability of the second layer of concealment, which can adjust the breath, hide the strength, and can hide from the monks of the same realm.

   However, monks on the sixth level of Qi Refining are still very common in the sect, and there is no need to hide their cultivation.

   If you want to go outside the range of the sect, you can use a disguise method recorded in "The Brief Introduction of Demon Cultivation" to change your appearance, so that even familiar people can't recognize him.

  Liu Yu thought about it for a long time, but still planned to ask that sister Jiang Qiushuijiang to see if she had any news of this kind of medicine pill.

   This Senior Sister Jiang has long sleeves and is good at dancing. She has a wide range of contacts in the door, and she is the most well-informed among the familiar colleagues.

   Running Mu Lingjue to transport Da Zhou Tianchun's mana, and completed today's meditation practice, Liu Yu left the cave.

   Leaving the cave, he made a gesture with both hands, blessed himself with the wind-fighting technique, and jumped lightly in the mountains and forests.

   In less than a quarter of an hour, he arrived in front of Jiang Qiushui's cave.

  Liu Yu took out a sound transmission from the storage bag, whispered a few words, and then turned it into white light and flew into the cave.

   The sound transmission was not blocked by the ban. Seeing this, he waited patiently.

   "Rumble"

The    stone gate moved slowly, Jiang Qiushui closed the restriction of the cave, and appeared in Liu Yu's sight.

   Maybe it was because she was cultivating in the mansion, she was dressed very casually, a white gauze covered her exquisite body, and her long black hair was casually draped over her shoulders.

   Maybe it was a problem with the material of the gauze, or maybe the cultivators had too good eyesight, and could faintly see the "dangerous peaks" through the gauze.

  Liu Yu moved his eyes away slightly, not daring to look more, and forcibly suppressed the restlessness in his heart.

  Jiang Qiushui was actually quite surprised by Liu Yu's arrival.

   The performance of this little junior brother in recent years is eye-catching. In just three years, he has improved his cultivation base by two levels, and he is also an alchemist. It is said that he has attracted the attention of the core figures of the other courtyard.

   She already regarded Liu Yu as a cultivator with a bright future, but she did not neglect her.

   Maybe it was because she had just finished practicing. At this time, her expression was a little lazy, no longer arrogant, and her voice was soft.

   "Junior Brother Liu is really a rare visitor, come and sit first!"

  Jiang Qiushui chuckled and led Liu Yu to the cave.

   Liu Yu followed closely, his eyes glanced at random, observing the decoration of the cave.

   Neat and clean, this is the first impression.

   Several pots of beautiful flowers and plants are placed in every corner of the cave, which is pleasing to the eye, and it seems that some incense is used, and there is a good smell in the air.

  The two came to the hall and sat opposite each other at a wooden table.

  Jiang Qiushui brushed a few strands of hair scattered in front of his forehead, pinned it to his ears, and then poured a cup of spiritual tea for Liu Yu before asking him about his purpose.

   "To tell the truth, I have reached the peak of the sixth level of Qi refining, and my current cultivation base has not made any progress for three months."

   "I took the liberty to disturb me this time. I just wanted to ask my sister, do you have any news about the breakthrough pill, or any news that is useful for breaking through the bottleneck?"

   Liu Yu took a sip of the spirit tea, only to feel the fragrance of his lips and teeth, and deliberately paused before saying the words he had prepared.

   But he concealed the fact that he was stuck in the bottleneck for eight months, and changed it to three months.

  Jiang Qiushui heard this, his bright eyes rolled at him, and he said angrily:

   "Breakthrough pills are extremely rare, and those on the Zongmen exchange list have already been exchanged."

   "As soon as this kind of medicinal pill is refined in the sect, most of them have already been reserved by fellow sects who have connections."

   "Besides, if there is news about the breakthrough pill, senior sister would have already bought it herself, and you don't need to be stuck in the late stage of qi refining for two years. Which round will get you!"

   "On the exchange list, there are still some medicinal pills that have a little or no breakthrough effect, but I don't recommend that junior and junior exchange them."

  Zongmen Contribution Point is a special currency in Yuanyang Zong, which has many uses within Zongmen, such as being able to exchange treasures on Zongmen's exchange list.

   There are many magical instruments, medicinal pills, etc. that are hard to find in the outside world on the    Zongmen exchange list, and even the foundation pills are available, which can only be exchanged with Zongmen contributions.

   However, it is worth mentioning that Zongmen contributions can be obtained through quests or by trading spirit stones with other disciples.

   Of course, when exchanging some strategic resources, there will be some restrictions, which need to be approved before exchanging.

   "Sister, don't worry, I know the benefits."

   Liu Yu nodded and said, but he was a little disappointed in his heart. He still underestimated the rarity of the breakthrough pills.

  I didn't get any useful information from Jiang Qiushui, and the other familiar colleagues were not as high as her, so I guess it was no use asking, so I had to find another way and think about it in the long run.

  Thinking about this, Liu Yu's interest has been greatly reduced, and he doesn't want to stay for a long time.

   Reluctantly chatted with Jiang Qiushui for a while, drank the cup of spirit tea, and praised a few words loudly:

   "Good tea, this tea is really good, the fragrance on the lips and teeth is unforgettable."

   "It's just that Junior Brother still has important things to do. Next time, Senior Sister will come here to taste spirit tea and say goodbye!"

   After speaking, Liu Yu bowed his hands to Jiang Qiushui, bid farewell, and turned to leave.

"Humph!"

   Jiang Qiushui could not understand the meaning of this junior brother, she hummed a little dissatisfied, but did not make a sound to hold back.
"""

# Extract entities
entities = extract_entities(chapter_text)

# Summarize the chapter
summary = summarize_chapter(chapter_text)

# Output results
print("Entities:", entities)
print("Chapter Summary:", summary)
