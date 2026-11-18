"use strict";

const SELECTED_WEIGHT = 7;
const ROUTE_WEIGHT = 4;
const BLIND_EXTRA_WEIGHT = 2;
const DIMMED_OPACITY = 0.45;
const FIT_PADDING_PX = [40, 40];
const RASTER_PANE = "rasters";
const RASTER_PANE_Z_INDEX = 250;   // under the vector pane (400), so routes stay on top of the layers

let bridge = null;
let map = null;
let tileLayer = null;
let tileLayerVisible = true;
let legend = null;
const markers = {};
const overlays = {};
let routeLayers = [];
let selectedIndex = -1;
let tilesFailed = 0;

function pointIcon(kind) {
  const letter = kind === "start" ? "S" : "E";
  return L.divIcon({ className: "", html: `<div class="point-marker point-${kind}">${letter}</div>`,
                     iconSize: [0, 0] });
}

function setPoint({ kind, lat, lon }) {
  if (markers[kind]) {
    markers[kind].setLatLng([lat, lon]);
    return;
  }
  const marker = L.marker([lat, lon], { draggable: true, icon: pointIcon(kind) }).addTo(map);
  marker.on("dragend", () => {
    const p = marker.getLatLng();
    bridge.markerMoved(kind, p.lat, p.lng);
  });
  markers[kind] = marker;
}

function clearPoints() {
  Object.values(markers).forEach((m) => m.remove());
  Object.keys(markers).forEach((k) => delete markers[k]);
}

function setTileSource({ template, attribution, maxNativeZoom, maxZoom }) {
  if (tileLayer) tileLayer.remove();
  tilesFailed = 0;
  tileLayer = L.tileLayer(template, { attribution, maxNativeZoom, maxZoom });
  if (tileLayerVisible) tileLayer.addTo(map);
  tileLayer.on("tileerror", () => { tilesFailed += 1; });
  tileLayer.on("load", () => bridge.tilesSettled(tilesFailed));
  map.setMaxZoom(maxZoom);
}

function setOverlay({ name, dataUrl, bounds, opacity, visible }) {
  if (overlays[name]) overlays[name].remove();
  const layer = L.imageOverlay(dataUrl, bounds, { opacity, interactive: false, pane: RASTER_PANE });
  overlays[name] = layer;
  if (visible) layer.addTo(map);
}

function clearOverlays() {
  Object.values(overlays).forEach((layer) => layer.remove());
  Object.keys(overlays).forEach((name) => delete overlays[name]);
}

function setOverlayVisible({ name, visible }) {
  const layer = overlays[name];
  if (!layer) return;
  if (visible) layer.addTo(map); else layer.remove();
}

function clearRoutes() {
  routeLayers.forEach((group) => group.forEach((layer) => layer.remove()));
  routeLayers = [];
  selectedIndex = -1;
  if (legend) { legend.remove(); legend = null; }
}

function showRoutes({ routes, blindColour }) {
  clearRoutes();
  const bounds = [];
  routes.forEach((route) => {
    const line = L.polyline(route.line, { color: route.colour, weight: ROUTE_WEIGHT, opacity: 0.95 }).addTo(map);
    line.on("click", () => bridge.routeSelected(route.index));
    const blind = route.blind.map((stretch) =>
      L.polyline(stretch, { color: blindColour, weight: ROUTE_WEIGHT + BLIND_EXTRA_WEIGHT, opacity: 1,
                            lineCap: "butt", interactive: false }).addTo(map));
    routeLayers.push([line, ...blind]);
    bounds.push(...route.line);
  });
  if (bounds.length) map.fitBounds(bounds, { padding: FIT_PADDING_PX });
  legend = L.control({ position: "bottomleft" });
  legend.onAdd = () => {
    const div = L.DomUtil.create("div", "legend");
    div.innerHTML = routes.map((r) =>
      `<div class="row" data-index="${r.index}"><span class="swatch" style="background:${r.colour}"></span>${r.label}</div>`
    ).join("") + `<div class="blind-note"><span class="swatch" style="background:${blindColour}"></span>blind stretch</div>`;
    return div;
  };
  legend.addTo(map);
  selectRoute({ index: 0 });
}

function selectRoute({ index }) {
  selectedIndex = index;
  routeLayers.forEach((group, i) => {
    const line = group[0];
    const selected = i === index;
    line.setStyle({ weight: selected ? SELECTED_WEIGHT : ROUTE_WEIGHT, opacity: selected ? 1 : DIMMED_OPACITY });
    if (selected) line.bringToFront();
    group.slice(1).forEach((layer) => {
      layer.setStyle({ weight: (selected ? SELECTED_WEIGHT : ROUTE_WEIGHT) + BLIND_EXTRA_WEIGHT,
                       opacity: selected ? 1 : DIMMED_OPACITY });
      if (selected) layer.bringToFront();
    });
  });
  document.querySelectorAll(".legend .row").forEach((row) =>
    row.classList.toggle("selected", Number(row.dataset.index) === index));
}

function setTileSourceVisible({ visible }) {
  tileLayerVisible = visible;
  if (!tileLayer) return;
  if (visible) tileLayer.addTo(map); else tileLayer.remove();
}

function setView({ lat, lon, zoom }) {
  map.setView([lat, lon], zoom);
}

const HANDLERS = { setPoint, clearPoints, setTileSource, setOverlay, setOverlayVisible, clearOverlays, clearRoutes, showRoutes,
                   selectRoute, setView, setTileSourceVisible };

function handleCommand(name, payload) {
  const handler = HANDLERS[name];
  if (!handler) throw new Error("unknown map command " + name);
  handler(JSON.parse(payload));
}

new QWebChannel(qt.webChannelTransport, (channel) => {
  bridge = channel.objects.bridge;
  map = L.map("map", { zoomControl: true });
  map.createPane(RASTER_PANE).style.zIndex = RASTER_PANE_Z_INDEX;
  map.on("click", (event) => bridge.mapClicked(event.latlng.lat, event.latlng.lng));
  bridge.command.connect(handleCommand);
  bridge.mapReady();
});
