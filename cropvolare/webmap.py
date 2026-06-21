"""
Standalone interactive web map (Leaflet via folium) from the aggregated field.

Produces a single self-contained HTML file - no server, opens straight from
disk, and the heatmap is base64-embedded so the file can be emailed as one
attachment. Shows the NDVI overlay, every photo location, and highlighted
markers for the worst areas.
"""

import base64

import folium

_STATUS_COLOR = {
    "healthy": "green",
    "moderate": "orange",
    "stressed": "orange",
    "severe": "red",
    "unknown": "gray",
}


def _embed_png(path):
    """Read a PNG and return a data: URI so the HTML stays self-contained."""
    with open(path, "rb") as f:
        data = base64.b64encode(f.read()).decode("ascii")
    return f"data:image/png;base64,{data}"


def build_webmap(feature_collection, grid, cells, problems, fieldmap_png,
                 out_path, overlay_opacity=0.6):
    """Write a standalone Leaflet HTML map to out_path."""
    meta = feature_collection.get("metadata", {})
    bbox = meta.get("bbox")
    if bbox:
        min_lon, min_lat, max_lon, max_lat = bbox
        center = [(min_lat + max_lat) / 2.0, (min_lon + max_lon) / 2.0]
    else:
        center = [0.0, 0.0]

    fmap = folium.Map(location=center, zoom_start=18, max_zoom=22,
                      tiles="OpenStreetMap", control_scale=True)

    # NDVI heatmap overlay
    if fieldmap_png and grid.get("bounds"):
        folium.raster_layers.ImageOverlay(
            image=_embed_png(fieldmap_png),
            bounds=grid["bounds"],
            opacity=overlay_opacity,
            name="NDVI heatmap",
            interactive=False,
        ).add_to(fmap)

    # every photo location
    photos = folium.FeatureGroup(name="Photos", show=False)
    for f in feature_collection.get("features", []):
        geom = f.get("geometry")
        if not geom:
            continue
        lon, lat = geom["coordinates"]
        p = f["properties"]
        color = _STATUS_COLOR.get(p.get("status"), "gray")
        folium.CircleMarker(
            location=[lat, lon], radius=4, color=color, fill=True,
            fill_opacity=0.8,
            popup=folium.Popup(
                f"<b>{p['filename']}</b><br>"
                f"NDVI: {p['mean_ndvi']}<br>status: {p['status']}",
                max_width=250),
        ).add_to(photos)
    photos.add_to(fmap)

    # problem markers on top
    problem_group = folium.FeatureGroup(name="Problem areas", show=True)
    for prob in problems:
        folium.Marker(
            location=[prob["lat"], prob["lon"]],
            icon=folium.Icon(color="red", icon="exclamation-sign"),
            popup=folium.Popup(
                f"<b>Problem #{prob['rank']}</b><br>"
                f"NDVI: {prob['mean_ndvi']} ({prob['status']})<br>"
                f"{prob['lat']:.6f}, {prob['lon']:.6f}<br>"
                f"photos: {prob['photo_count']}",
                max_width=250),
        ).add_to(problem_group)
    problem_group.add_to(fmap)

    folium.LayerControl(collapsed=False).add_to(fmap)

    if bbox:
        fmap.fit_bounds([[min_lat, min_lon], [max_lat, max_lon]])

    fmap.save(out_path)
