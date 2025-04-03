import streamlit as st
import folium
import numpy as np
from streamlit_folium import st_folium

def interpolate_points(points, stappen=100):
    coords = []
    for i in range(len(points) - 1):
        lon_start, lat_start = points[i]
        lon_eind, lat_eind = points[i+1]
        lons = np.linspace(lon_start, lon_eind, stappen)
        lats = np.linspace(lat_start, lat_eind, stappen)
        for lat, lon in zip(lats, lons):
            coords.append((lat, lon))
    return coords

# Belangrijke punten (longitude, latitude)
southampton = (-1.4044, 50.9097)
cherbourg   = (-1.62, 49.65)
queenstown  = (-8.3, 51.85)
sinking     = (-49.9, 41.7)  
new_york    = (-74.0060, 40.7128)

actual_points = [southampton, cherbourg, queenstown, sinking]
planned_points = [sinking, new_york]

actual_route = interpolate_points(actual_points, stappen=100)
planned_route = interpolate_points(planned_points, stappen=100)

# Maak de kaart
m = folium.Map(location=[48, -40], zoom_start=3)

# Werkelijke route (blauw)
folium.PolyLine(actual_route, color="blue", weight=3, opacity=1).add_to(m)

# Geplande route (gestippeld rood)
folium.PolyLine(planned_route, color="red", weight=3, opacity=1, dash_array="10, 10").add_to(m)

# Opstaplocaties markers
stops = [
    {"name": "Southampton", "coords": [southampton[1], southampton[0]]},
    {"name": "Cherbourg",   "coords": [cherbourg[1], cherbourg[0]]},
    {"name": "Queenstown",  "coords": [queenstown[1], queenstown[0]]}
]

for stop in stops:
    folium.Marker(
        stop["coords"],
        popup=stop["name"],
        icon=folium.Icon(color="green", icon="anchor", prefix="fa")
    ).add_to(m)

# Marker voor het ijsbergincident
folium.Marker(
    [sinking[1], sinking[0]],
    popup="IJsberg incident",
    icon=folium.CustomIcon("iceberg.png", icon_size=(40, 40))
).add_to(m)

# Marker voor de Titanic (bij benadering iets naast de ijsberg)
folium.Marker(
    [sinking[1] + 1, sinking[0] + 4],
    popup="Titanic",
    icon=folium.CustomIcon("titanic.png", icon_size=(40, 40))
).add_to(m)

# Marker voor de geplande bestemming (New York)
folium.Marker(
    [new_york[1], new_york[0]],
    popup="Geplande bestemming: New York",
    icon=folium.Icon(color="blue", icon="flag")
).add_to(m)

# Streamlit pagina
st.title("Titanic Route met IJsberg Incident")
st.markdown("""
Deze interactieve kaart toont de route die de Titanic zou hebben gevaren. 
- De **blauwe lijn** geeft de werkelijke route weer.
- De **gestippelde rode lijn** geeft de geplande route (van de zinklocatie naar New York) weer.
- De havens en belangrijke punten worden gemarkeerd.
""")

# Geef de folium kaart weer in de Streamlit app
st_data = st_folium(m, width=700, height=500)
