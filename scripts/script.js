const services = [
  "LULC (Unimatch V2)",
  "Brickfield (Unimatch V2)",
  "LULC (Unet)",
  "Brickfield"
  // "NDVI",
  // "NDWI",
  // "Building Categorization",
  // "Water Detection",
  // "Green Detection",
  // "Building Detection",
];

const serviceSelector = $("#service-select");
const districtSelector = $("#district-select");
const upazilaSelector = $("#upazila-select");

let map = false;
let rect = false;

function getBounds() {
  let textValues = [];
  $("input[type=number]").each(function () {
    const v = $(this).val();
    textValues.push(v);
  });

  let topLeftNotFilled = textValues[0].length == 0 || textValues[1].length == 0;
  let bottomRightNotFilled =
    textValues[2].length == 0 || textValues[3].length == 0;

  return topLeftNotFilled || bottomRightNotFilled
    ? false
    : {
        topLeft: [textValues[0], textValues[1]],
        bottomRight: [textValues[2], textValues[3]],
      };
}

function getAdmins() {
  return {
    district: districtSelector.val(),
    upazila: upazilaSelector.val(),
  };
}

function populateUpazilaSelector(e) {
  // console.debug($(e).val())
  const k = $(e).val();
  if (k.length == 0) return false;

  upazilaSelector
    .empty()
    .append('<option selected="selected" value="">Select Upazila</option>');
  Object.keys(admin_zones[k]).forEach((x) =>
    upazilaSelector.append($("<option>").attr("value", x).text(x))
  );
}

function onSubmit() {
  // const r = {
  //     service: serviceSelector.val(),
  //     bbox_string: rect ? rect.getBounds().toBBoxString() : '',
  //     ...getAdmins()
  // }

  const r = $("form")
    .serializeArray()
    .reduce((r, e) => {
      r[e.name] = e.value;
      return r;
    }, {});

  $("#output").text(JSON.stringify(r, null, 4));

  return false;
}

function clearRectangle() {
  if (!rect) return;

  rect.remove();
  rect = false;
}

function clearRectangleInputs() {
  $("input").each(function () {
    $(this).val("");
  });
}

function updateRectangle() {
  inputValues = getBounds();

  if (!inputValues) return;

  // define rectangle geographical bounds
  let bounds = [inputValues.topLeft, inputValues.bottomRight];

  if (rect) clearRectangle();

  // create an orange rectangle
  rect = L.rectangle(bounds, { color: "#ff7800", weight: 1 }).addTo(map);

  // making sure the corner values are in correct order
  let vals = rect.getBounds().toBBoxString().split(',')
  // $(`#topLeft-lattitude`).val(vals[0]);
  // $(`#topLeft-longitude`).val(vals[1]);
  // $(`#bottomRight-lattitude`).val(vals[2]);
  // $(`#bottomRight-longitude`).val(vals[3]);
  

  // zoom the map to the rectangle bounds
  map.fitBounds(bounds);
}

function clickRectangleCornersOnMap() {
  let k = 0;
  $("#selection-rect").prop("checked", true);
  const buttonRectangle = document.querySelector('#click-draw-rectangle')
  
  if (k == 0) {
    buttonRectangle.disabled = true
  }
  
  k_list = ["topLeft", "bottomRight"];
  map.on("click", function (e) {
    // alert("Lat, Lon : " + e.latlng.lat + ", " + e.latlng.lng)
    // box[k] = [e.latlng.lat, e.latlng.lng]
    
    $(`#${k_list[k]}-lattitude`).val(e.latlng.lat);
    $(`#${k_list[k]}-longitude`).val(e.latlng.lng);
    
    if (k != 0) {
      buttonRectangle.disabled = false
    }
    
    
    // if (k == 0) {
    //   $(`#${k_list[k+1]}-lattitude`).val(e.latlng.lat);
    //   $(`#${k_list[k]}-longitude`).val(e.latlng.lng);
    // } else if (k == 1) {
    //   $(`#${k_list[k-1]}-lattitude`).val(e.latlng.lat);
    //   $(`#${k_list[k]}-longitude`).val(e.latlng.lng);
    // }

    k += 1;

    if (k >= 2) {
      map.off("click");
      updateRectangle();
    }
  });
}

function resetForm(){
    $('form').trigger('reset');
    clearRectangle();
    map.panTo(new L.LatLng(23.7806365,90.4193257));
}

try {
  services.forEach((x) =>
    serviceSelector.append($("<option>").attr("value", x).text(x))
  );
  Object.keys(admin_zones).forEach((x) =>
    districtSelector.append($("<option>").attr("value", x).text(x))
  );

  map = makeMap();
} catch (err) {
  console.error(err);
}
