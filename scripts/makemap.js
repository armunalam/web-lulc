let markers = [];

function makeMap(vectorObject=false) {
  let map = L.map("map", {
    center: [23.7806365,90.4193257],
    minZoom: 1,
    zoom: 7,
    // zoomSnap: 0,
    // worldCopyJump: true,
    // crs: 'ESRI:54030'
  });

  L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
    attribution:
      '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
  }).addTo(map);

  L.control.scale().addTo(map);

  function getColor() {
    let c = "grey";

    return c;
  }

  function style(feature) {
    return {
      weight: 1,
      opacity: 1,
      color: "black",
      dashArray: "1",
      fillOpacity: 0.3,
      fillColor: "green",
    };
  }

  function highlightFeature(e) {
    var layer = e.target;

    // layer.setStyle({
    //     weight: 5,
    //     color: '#666',
    //     dashArray: '',
    //     fillOpacity: 0.7
    // });

    // if (!L.Browser.ie && !L.Browser.opera && !L.Browser.edge) {
    //     layer.bringToFront();
    // }

    info.update(layer.feature.properties);
    // console.log(layer.feature.properties);
  }

  function onEachFeature(feature, layer) {}


//   map.setMaxBounds(geojson.getBounds().pad(0.1))//.setView();
  // var group = new L.featureGroup(markers);

  // map.fitBounds(group.getBounds());

  return map;
}
