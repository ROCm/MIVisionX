function openNav() {
    document.getElementById("mySidenav").style.width = "250px";
    document.getElementById("main").style.marginLeft = "250px";
}

function closeNav() {
    document.getElementById("mySidenav").style.width = "0";
    document.getElementById("main").style.marginLeft = "0";
}

function myreload() {
    location.reload();
}



function insertIntoTopKTable() {
    var table = document.getElementById("top-5-match-table");
    var row = table.insertRow(-1);
    var row2 = table.insertRow(-1);
    data.topKStats.forEach(function (element) {
        var cell = row.insertCell(-1);
        cell.innerHTML = "<b>" + element.matches + "</b>";
        var cell2 = row2.insertCell(-1);
        cell2.innerHTML = "<b>" + element.accuracyPer.toFixed(2) + " %</b>";
    });
}

function showStatsOnTable() {
    $('b[id ^= "stat"]').each(function (element) {
        var property = this.id.split("_")[1];
        var round = Number(this.getAttribute('data-round'));
        if (data.stats[property])
            this.innerHTML = data.stats[property].toFixed(round);

    });
}


function loadLabelSummary(labelSummaryLocal) {
    var myTableBody = $('#label-summary-table').find("tbody");
    myTableBody.empty();
    labelSummaryLocal.forEach(function (label) {
        var row = $('<tr>');
        var id = label.id;
        var totalImageClass = label.totalImages > 0 ? 'color-green' : '';

        var misclassifiedClass = label.misclassifiedTop1 == 0 ? 'color-green' : (label.totalImages > 0 ? 'color-red' : 'color-black');


        row.append($('<td>').attr({
            'class': 'blue filter-image',
            'data-id': label.id
        }).text(label.id));

        row.append($('<td>').attr({
            'class': 'blue left-align filter-image',
            'data-id': label.id
        }).text(label.label));

        row.append($('<td>').attr('class', 'blue ' + totalImageClass).text(label.totalImages));
        row.append($('<td>').attr('class', 'blue').text(label.matchedTop1Per.toFixed(2)));
        row.append($('<td>').attr('class', 'blue').text(label.matchedTop5Per.toFixed(2)));
        row.append($('<td>').attr('class', 'blue').text(label.match1));
        row.append($('<td>').attr('class', 'blue').text(label.match2));
        row.append($('<td>').attr('class', 'blue').text(label.match3));
        row.append($('<td>').attr('class', 'blue').text(label.match4));
        row.append($('<td>').attr('class', 'blue').text(label.match4));
        row.append($('<td>').attr({
            'class': 'blue filter-image ' + misclassifiedClass,
            'data-id': label.id,
            'data-type': 'misclassified'
        }).text(label.misclassifiedTop1));

        row.append($('<td>').html('<input id="id_"' + id + ' name="id[' + id + ']" type="checkbox" value="' + id + '" onClick="highlightRow(this);"></input>'));
        myTableBody.append(row);
    });


}

function filterLabelTable(rowNum, Datavar) {
    var labelSummaryFiltered = [];
    var allEmpty = true;
    $('input[id ^= "fl_"]').each(function (element) {
        var property = this.id.split("_")[1];
        var value = this.value;
        var compare = this.getAttribute('data-compare');
        if (value) {
            allEmpty = false;
            labelSummary.forEach(function (label) {
                var compareResult = false;

                if (compare == 'contains') {
                    compareResult = label[property].toLowerCase().includes(value.toLowerCase());

                } else {
                    compareResult = label[property] == value;
                }
                if (compareResult) {
                    labelSummaryFiltered.push(label);
                }
            });
        }
    });
    if (allEmpty) {
        loadLabelSummary(labelSummary);
    } else {
        loadLabelSummary(labelSummaryFiltered);
    }
}

function clearLabelFilter() {
    $('input[id ^= "fl_"]').val('');

    loadLabelSummary(labelSummary);
}

function highlightRow(element) {
    var parentRow = element.parentElement.parentElement;
    parentRow.classList.toggle("highlight-row");
    console.log(element.parentElement.parentElement);

}


//Create copy of imageSummary
var imageSummaryFiltered = JSON.parse(JSON.stringify(imageSummary));

function clearResultFilter() {
    $('input[id ^= "fli_"]').val('');
    imageSummaryFiltered = JSON.parse(JSON.stringify(imageSummary));
    loadImageResults(imageSummaryFiltered);
    updateItems();
}

function filterResultTable(e) {
    imageSummaryFiltered = [];
    var allEmpty = true;
    if (e) {
        var code = (e.keyCode ? e.keyCode : e.which);

    }
    var combinationOp = $("input:radio[name='filterType']:checked").val();
    var notOp = $('#not-op').is(":checked");


    var filters = {};
    var count = 0;

    // if (combinationOp == 'and')
    //     filterSource = imageSummaryFiltered
    // else
    //     filterSource = imageSummary

    $('input[id ^= "fli_"]').each(function (element) {
        var allNames = this.id.split("_");
        var property = allNames[1];
        var property_index = null;
        var prop_name_new = allNames[1];

        if (allNames[2]) {
            property_index = parseInt(allNames[2]);
            prop_name_new = prop_name_new + '_' + property_index;
        }
        var value = this.value;
        var compare = this.getAttribute('data-compare');
        var isArrayProp = this.getAttribute('data-is-array');

        var op = $('#fli_op_' + prop_name_new).val();

        // if (!op) {
        //     op = 'eq';
        // }

        if (value) {
            allEmpty = false;
            filters[this.id] = []
            // if (compare)
            //     filters[this.id].push('includes');
            // else
            //     filters[this.id].push('eq');
            filters[this.id].push(op);

            filters[this.id].push(value)

            // imageSummary.forEach(function (label) {

            //     var compareResult = false;
            //     var propertyValue = label[property];
            //     if (isArrayProp) {
            //         propertyValue = propertyValue[property_index];
            //     }
            //     if (compare == 'contains') {
            //         compareResult = propertyValue.toLowerCase().includes(value.toLowerCase());
            //     } else {
            //         compareResult = propertyValue == value;
            //     }
            //     if (compareResult) {
            //         imageSummaryFiltered.push(label);
            //     }
            // });
        }





    });

    console.log(filters);

    imageSummaryFiltered = imageSummary.filter(function (item) {
        // return item['gt'] == 1;
        var allTrue = true;
        var anyTrue = false;

        for (var key in filters) {
            // console.log(key)
            var allNames = key.split("_");
            var property = allNames[1];
            var property_index = null;
            if (allNames[2]) {
                property_index = parseInt(allNames[2]);
            }
            var itemValue = item[property];
            if (allNames[2] >= 0) {
                itemValue = itemValue[property_index];
            }

            if (!compare(itemValue, filters[key][1], filters[key][0])) {
                allTrue = false;
            } else {
                anyTrue = true;
            }


        }

        if (notOp == true) {
            anyTrue = !anyTrue;
            allTrue = !allTrue;
        }
        if (combinationOp == 'or') {
            return anyTrue;
        } else if (combinationOp == 'and') {
            return allTrue
        }

        return true;

    });

    if (allEmpty) {
        var perPage = 40;
        var pageNumber = 1;
        imageSummaryFiltered = JSON.parse(JSON.stringify(imageSummary));
        var showFrom = perPage * (pageNumber - 1);
        var showTo = showFrom + perPage;
        loadImageResults(imageSummaryFiltered.slice(showFrom, showTo));

    } else {
        loadImageResults(imageSummaryFiltered);
    }
    updateItems();
}



function loadImageResults(imageSummaryLocal) {
    var myTableBody = $('#image-results-table').find("tbody");
    myTableBody.empty();
    var count = 0;
    imageSummaryLocal.forEach(function (im) {

        if (count > 40)
            return;

        count++;

        var row = $('<tr>');
        var id = im.imgName;
        var matchClass = im.match > 0 ? 'color-green' : 'color-red';
        // var imgLink = $('a').attr('href', im.filePath).attr('target', '_blank').text(im.imageName);

        var imgHtml = '<img onclick="showModal(event)" src="' +
            im.filePath +
            '" data-alt="<b>GROUND TRUTH:</b>' +
            im.gtText + '<br><b>CLASSIFIED AS:</b>' +
            im.labelTexts[0] + '" width="30" height="30">';
        var imgLinkHtml = '<a href="' + im.filePath + '" target="_blank">' + im.imageName + '</a>';

        row.append($('<td>').attr('class', 'imgcell').html(imgHtml));
        row.append($('<td>').html(imgLinkHtml));
        row.append($('<td>').attr('class', 'imgcell left-align').text(im.gtText));
        row.append($('<td>').attr('class', 'imgcell').text(im.gt));

        for (var i = 0; i < 5; i++) {
            row.append($('<td>').attr('class', 'imgcell').text(im.labels[i]));
        }
        row.append($('<td>').attr('class', 'imgcell bold ' + matchClass).text(im.match));

        for (i = 0; i < 5; i++) {
            row.append($('<td>').attr('class', 'imgcell left-align').text(im.labelTexts[i]));
        }

        for (i = 0; i < 5; i++) {
            row.append($('<td>').attr('class', 'imgcell').text(im.probs[i].toFixed(4)));
        }
        myTableBody.append(row);
    });


}

function showModal(event) {
    console.log(event.target);
    var modal = document.getElementById('myModal');
    var modalImg = document.getElementById('img01');
    var modalCaption = document.getElementById('caption');
    modal.style.display = "block";
    modalCaption.innerHTML = event.target.dataset.alt;
    modalImg.src = event.target.src;
}

function closeModal() {
    var modal = document.getElementById('myModal');
    modal.style.display = "none";

}

// We'll create a function to update the paginator
// whenever we add or remove items from it.
// You'll need to ensure that this function is called
// whenever your items dynamically change.
function updateItems() {
    var $paginator = $(".pagination-page");
    // Notice that we're not using `var items = ...` -
    // instead we're using `items` variable declared outside this function.
    items = imageSummaryFiltered;
    // $("#image-results-table tbody tr");
    // We'll update the number of items that the pagination element expects
    // using a method from the simplePagination documentation: `updateItems`.
    $paginator.pagination("updateItems", items.length);
    // Next, we'll re-select the page so the `onPageClick` function we already
    // programmed will handle hiding and showing the correct content for us.

    // Note that the page we were on before may no longer exist
    // (i.e. if elements were removed and that page is no longer required),
    // so we will make sure to select an existing page.
    var page = Math.min(
        $paginator.pagination("getCurrentPage"),
        $paginator.pagination("getPagesCount")
    );

    $('#table-item-count').html('Total Items: ' + items.length);
    $paginator.pagination("selectPage", page);
}


jQuery(function ($) {
    // Consider adding an ID to your table
    // incase a second table ever enters the picture.
    var items = imageSummaryFiltered; //$("#image-results-table tbody tr");

    var numItems = items.length;
    var perPage = 40;
    // Only show the first 2 (or first `per_page`) items initially.
    // items.slice(perPage).hide();
    // Now setup the pagination using the `.pagination-page` div.
    $(".pagination-page").pagination({
        items: numItems,
        itemsOnPage: perPage,
        cssStyle: "light-theme",

        // This is the actual page changing functionality.
        onPageClick: function (pageNumber) {
            // We need to show and hide `tr`s appropriately.
            var showFrom = perPage * (pageNumber - 1);
            var showTo = showFrom + perPage;
            loadImageResults(imageSummaryFiltered.slice(showFrom, showTo));

            // // We'll first hide everything...
            // items.hide()
            //     // ... and then only show the appropriate rows.
            //     .slice(showFrom, showTo).show();
        }
    });




    // We'll call this now so the initial items can be loaded.
    updateItems();

    // EDIT: Let's cover URL fragments (i.e. #page-3 in the URL).
    // More thoroughly explained (including the regular expression) in: 
    // https://github.com/bilalakil/bin/tree/master/simplepagination/page-fragment/index.html

    // We'll create a function to check the URL fragment
    // and trigger a change of page accordingly.
    function checkFragment() {
        // If there's no hash, treat it like page 1.
        var hash = window.location.hash || "#page-1";

        // We'll use a regular expression to check the hash string.
        hash = hash.match(/^#page-(\d+)$/);

        if (hash) {
            // The `selectPage` function is described in the documentation.
            // We've captured the page number in a regex group: `(\d+)`.
            $(".pagination-page").pagination("selectPage", parseInt(hash[1]));
        }
    }

    // We'll call this function whenever back/forward is pressed...
    $(window).bind("popstate", checkFragment);

    // ... and we'll also call it when the page has loaded
    // (which is right now).
    checkFragment();



});





function loadScorings() {
    var myTableBody = $('#standard-scoring-table').find("tbody");
    myTableBody.empty();
    var row = $('<tr>');
    data.scores.matchCounts.forEach(function (value) {
        row.append($('<td>').text(value));
    });
    myTableBody.append(row);

    row = $('<tr>');
    data.scores.modelScores.forEach(function (value) {
        row.append($('<td>').text(value.toFixed(2) + "%"));
    });
    myTableBody.append(row)


}


function compare(a, b, operator) {

    switch (operator) {
        case 'contains':
            return a.toString().toLowerCase().includes(b.toString().toLowerCase());
            break;
        case 'eq':
            return a == b;
        case 'neq':
            return a != b;
        case 'gt':
            return a > b;
        case 'lt':
            return a < b;
        case 'gte':
            return a >= b;
        case 'lte':
            return a <= b;
        default:
            return a == b;
    }
}

function sortImageResults(fieldName, direction) {
    var fieldNameArr = fieldName.split('_');
    var propertyName = fieldNameArr[0];
    var propertyIndex;
    if (fieldNameArr[1]) {
        propertyIndex = parseInt(fieldNameArr[1]);
    }
    imageSummaryFiltered.sort(function (a, b) {
        var aVal = a[propertyName];
        var bVal = b[propertyName];
        if (fieldNameArr[1]) {
            aVal = aVal[propertyIndex];
            bVal = bVal[propertyIndex];
        }
        var result = (aVal < bVal) ? 1 : -1;
        if (direction == -1) {
            result = result * -1;
        }
        return result;
    });
    loadImageResults(imageSummaryFiltered);
}



document.addEventListener('DOMContentLoaded', function () {
    $('#generation-date').html(data.summaryGenerationDate);

    insertIntoTopKTable();
    showStatsOnTable();
    loadLabelSummary(labelSummary);
    loadScorings();

    $('.filter-image').on('click', function (event) {
        var element = event.target;
        var targetValue = element.dataset.id;
        var toMisclassified = element.dataset.type;
        console.log(element)
        console.log(targetValue)
        console.log(toMisclassified)
        $('input[id ^= "fli_"]').val('');

        if (!toMisclassified) {
            $('#fli_gt').val(targetValue)
        } else {
            $('#fli_labels_0').val(targetValue);
            $('#fli_match').val(1);
            $('#fli_op_match').val('neq')
        }
        filterResultTable(event);
        window.location.href = '#table4';

    });

    $('#not-op').click(function () {
        filterResultTable();
    });
    $("input:radio[name='filterType']").click(function () {
        filterResultTable();
    });
    // loadImageResults(imageSummary);

    $('select[id ^= "fli_op_"]').change(function () {
        filterResultTable();
    })


    $('.sort-field').click(function (event) {
        var targetField = $(this).data('field');
        var direction = $(this).data('sort-dir');

        $('.sort-field span').html('');

        direction = direction ? parseInt(direction) : 1;
        sortImageResults(targetField, direction);
        direction = direction * -1;
        $(this).data("sort-dir", direction);
        var $this = $(this).find('span');
        $this.html(direction == -1 ? '&nbsp;&#x25B4;' : '&nbsp;&#x25BE;');

    });
});