from re import sub
import streamlit as st
import requests
from config import API_URL
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np
import plotly.graph_objects as go


def app():
    st.title("Text Scoring Engine")

    st.markdown("""
                Here, you can evaluate the scoring framework on either the ASAP-AES scoring dataset. Select an essay prompt
                below and the system will return the predicted score. Note that the training set is not yet fully extensive and this system
                is not yet production-ready.
                """)

    essay_set = st.radio("Select the essay set", [
                         "Set 3", "Set 4", "Set 5", "Set 6", "Practice A", "Practice B"])

    if essay_set == 'Set 3':
        st.markdown('''
                    FORGET THAT OLD SAYING ABOUT NEVER taking candy from strangers. No, a better piece of advice for the solo cyclist would be, “Never accept travel advice from a collection of old-timers who haven’t left the confines of their porches since Carter was in office.” It’s not that a group of old guys doesn’t know the terrain. With age comes wisdom and all that, but the world is a fluid place. Things change.
At a reservoir campground outside of Lodi, California, I enjoyed the serenity of an early-summer evening and some lively conversation with these old codgers. What I shouldn’t have done was let them have a peek at my map. Like a foolish youth, the next morning I followed their advice and launched out at first light along a “shortcut” that was to slice away hours from my ride to Yosemite National Park.
They’d sounded so sure of themselves when pointing out landmarks and spouting off towns I would come to along this breezy jaunt. Things began well enough. I rode into the morning with strong legs and a smile on my face. About forty miles into the pedal, I arrived at the first “town.” This place might have been a thriving little spot at one time—say, before the last world war—but on that morning it fit the traditional definition of a ghost town. I chuckled, checked my water supply, and moved on. The sun was beginning to beat down, but I barely noticed it. The cool pines and rushing rivers of Yosemite had my name written all over them.
Twenty miles up the road, I came to a fork of sorts. One ramshackle shed, several rusty pumps, and a corral that couldn’t hold in the lamest mule greeted me. This sight was troubling. I had been hitting my water bottles pretty regularly, and I was traveling through the high deserts of California in June.
I got down on my hands and knees, working the handle of the rusted water pump with all my strength. A tarlike substance oozed out, followed by brackish water feeling somewhere in the neighborhood of two hundred degrees. I pumped that handle for several minutes, but the water wouldn’t cool down. It didn’t matter. When I tried a drop or two, it had the flavor of battery acid.
The old guys had sworn the next town was only eighteen miles down the road. I could make that! I would conserve my water and go inward for an hour or so—a test of my inner spirit.
Not two miles into this next section of the ride, I noticed the terrain changing. Flat road was replaced by short, rolling hills. After I had crested the first few of these, a large highway sign jumped out at me. It read: ROUGH ROAD AHEAD: DO NOT EXCEED POSTED SPEED LIMIT.
The speed limit was 55 mph. I was doing a water-depleting 12 mph. Sometimes life can feel so cruel.
I toiled on. At some point, tumbleweeds crossed my path and a ridiculously large snake—it really did look like a diamondback—blocked the majority of the pavement in front of me. I eased past, trying to keep my balance in my dehydrated state.
The water bottles contained only a few tantalizing sips. Wide rings of dried sweat circled my shirt, and the growing realization that I could drop from heatstroke on a gorgeous day in June simply because I listened to some gentlemen who hadn’t been off their porch in decades, caused me to laugh.
It was a sad, hopeless laugh, mind you, but at least I still had the energy to feel sorry for myself. There was no one in sight, not a building, car, or structure of any kind. I began breaking the ride down into distances I could see on the horizon, telling myself that if I could make it that far, I’d be fi ne.
Over one long, crippling hill, a building came into view. I wiped the sweat from my eyes to make sure it wasn’t a mirage, and tried not to get too excited. With what I believed was my last burst of energy, I maneuvered down the hill.
In an ironic twist that should please all sadists reading this, the building—abandoned years earlier, by the looks of it—had been a Welch’s Grape Juice factory and bottling plant. A sandblasted picture of a young boy pouring a refreshing glass of juice into his mouth could still be seen.
I hung my head.
That smoky blues tune “Summertime” rattled around in the dry honeycombs of my deteriorating brain.
I got back on the bike, but not before I gathered up a few pebbles and stuck them in my mouth. I’d read once that sucking on stones helps take your mind off thirst by allowing what spit you have left to circulate. With any luck I’d hit a bump and lodge one in my throat.
It didn’t really matter. I was going to die and the birds would pick me clean, leaving only some expensive outdoor gear and a diary with the last entry in praise of old men, their wisdom, and their keen sense of direction. I made a mental note to change that paragraph if it looked like I was going to lose consciousness for the last time.
Somehow, I climbed away from the abandoned factory of juices and dreams, slowly gaining elevation while losing hope. Then, as easily as rounding a bend, my troubles, thirst, and fear were all behind me.
GARY AND WILBER’S FISH CAMP—IF YOU WANT BAIT FOR THE BIG ONES, WE’RE YOUR BEST BET!
“And the only bet,” I remember thinking.
As I stumbled into a rather modern bathroom and drank deeply from the sink, I had an overwhelming urge to seek out Gary and Wilber, kiss them, and buy some bait—any bait, even though I didn’t own a rod or reel.
An old guy sitting in a chair under some shade nodded in my direction. Cool water dripped from my head as I slumped against the wall beside him.
“Where you headed in such a hurry?”
“Yosemite,” I whispered.
“Know the best way to get there?”
I watched him from the corner of my eye for a long moment. He was even older than the group I’d listened to in Lodi.
“Yes, sir! I own a very good map.”
And I promised myself right then that I’d always stick to it in the future.
“Rough Road Ahead” by Joe Kurmaskie, from Metal Cowboy, copyright © 1999 Joe Kurmaskie.
''')

    response = st.text_area("Enter the relevant segment of the report here.")

    submit = st.button("Submit")

    if submit:
        if 'Set' in essay_set:
            with st.spinner('Working'):
                score = requests.post(f"{API_URL}/predict-asap-aes/", json={
                                      "text": response, "essay_set": "".join(essay_set.lower().split())}).json()
                st.success(f"Your score is: {str(score['eval_score'])} / 4")
                chart = st.progress((score['eval_score']/4) * 1)
                if score['eval_score'] >= 3:
                    st.balloons()
        elif 'Practice' in essay_set:
            with st.spinner('Working'):
                score = requests.post(f"{API_URL}/predict-report/", json={
                                      "text": response, "essay_set": "-".join(essay_set.lower().split())}).json()
                st.success(f"Your score is: {str(score['eval_score'])} / 4")
                chart = st.progress((score['eval_score']/4) * 1)
                if score['eval_score'] >= 3:
                    st.balloons()

        fig, ax = plt.subplots()
        attentions = requests.get(API_URL + '/word-level-attention').json()
        print(attentions)

        key_phrase = st.selectbox(
            "Select a key phrase to view attention heatmap", score["keywords"])

        key_phrase_index = score["keywords"].index(key_phrase)
        # fig = px.imshow(np.array(attentions[0][key_phrase_index])[:len(key_phrase.split()),:len(score['text'].split())], width=500, height=2000)
        fig = go.Figure(data=go.Heatmap(z=np.array(attentions[0][key_phrase_index])[:len(
            key_phrase.split()), :len(score['text'].split())], x=score['text'].split(), y=key_phrase.split()))
        # fig.layout.height = 400
        # fig.layout.width = 600
        fig.update(layout_coloraxis_showscale=False)
        fig.layout.height = 600
        fig.layout.width = 800
        st.write(fig)
    # ax = sns.heatmap(attentions[0][0], cmap='viridis', xticklabels=False, yticklabels=False, ax=ax)
    # st.write(fig)
