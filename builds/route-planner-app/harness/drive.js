// Replays a recorded plan into the map page, as MainWindow would, then selects the last route.
(async () => {
  const p = await (await fetch("payloads.json")).json();
  const params = new URLSearchParams(location.search);
  const layers = (params.get("layers") || "").split(",");
  sendCommand("setTileSource", p.tiles);
  sendCommand("setView", { lat: 49.91, lon: 23.905, zoom: 14 });
  sendCommand("setPoint", { kind: "start", lat: p.start[0], lon: p.start[1] });
  sendCommand("setPoint", { kind: "end", lat: p.end[0], lon: p.end[1] });
  p.overlays.forEach((o) => sendCommand("setOverlay", { ...o, visible: layers.includes(o.name) }));
  sendCommand("showRoutes", p.routes);
  sendCommand("selectRoute", { index: p.routes.routes.length - 1 });
  if (params.get("probe")) {
    map.fire("click", { latlng: L.latLng(49.9, 23.9) });
    markers.start.fire("dragend");
    routeLayers[0][0].fire("click");
    const out = document.createElement("pre");
    out.id = "sent";
    out.textContent = JSON.stringify(window.sent);
    document.body.appendChild(out);
  }
  document.title = "driven";
})();
