import streamlit as st
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
import graphviz

pokemon=pd.read_csv("Pokemon.csv")
combats=pd.read_csv("combats.csv")

###pokemon seçim
col1,col2,col3=st.columns(3)

with col1:
    poke1=st.selectbox("İlk pokemon",pokemon["Name"])
    isim=poke1.lower()
    isim1 = poke1
    isim="images/"+isim+".png"
    st.image(isim)
    poke1=int(pokemon[pokemon["Name"]==poke1]["#"])

with col2:
    st.image("https://www.nicepng.com/png/full/271-2712237_vs-rooster-teeth.png")

with col3:
    poke2=st.selectbox("Ikinci pokemon",pokemon["Name"])
    isim=poke2.lower()
    isim2=poke2
    isim="images/"+isim+".png"
    st.image(isim)
    poke2=int(pokemon[pokemon["Name"]==poke2]["#"])

###winner dönüşüm
combats["Winner"]=combats["First_pokemon"]==combats["Winner"]
combats["Winner"]=np.where(combats["Winner"]==True,0,1)

###model ve skor
y=combats[["Winner"]]
x=combats[["First_pokemon","Second_pokemon"]]
tree=DecisionTreeClassifier()
model=tree.fit(x,y)
skor=model.score(x,y)
tahmin=model.predict([[poke1,poke2]])

kol1,kol2,kol3=st.columns(3)

with kol1:
    st.write("")

with kol2:
    buton=st.button("Başlat")
    if buton:
        if tahmin==0:
            st.header("Kazanan")
            st.subheader(isim1)
        else:
            st.header("Kazanan")
            st.subheader(isim2)
        st.balloons()
        st.graphviz_chart(export_graphviz((tree)))
