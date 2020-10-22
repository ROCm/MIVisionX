function sum(a, b) {
    return a + b;
}

function drawChart() {

    var chartData = google.visualization.arrayToDataTable([
        ['  ', 'Match', 'Mismatch', 'No Label'],
        ['Summary', data.stats.passCount, data.stats.totalMismatch, data.stats.totalNoGroundTruth]
    ]);
    var options = {
        title: 'Overall Result Summary',
        vAxis: {
            title: 'Images'
        },
        width: 800,
        height: 400
    };
    var chart = new google.charts.Bar(document.getElementById('Model_Stats'));
    chart.draw(chartData, google.charts.Bar.convertOptions(options));
}


function drawTopKResultChart() {
    var chartData = new google.visualization.DataTable();
    chartData.addColumn('string', 'Top K');
    chartData.addColumn('number', 'Matchs');
    chartData.addRows([
        ['Matched Top5 Choice', data.topCounts.reduce(sum, 0)],
        ['MisMatched', data.stats.totalMismatch]
    ]);
    var options = {
        title: 'Image Match/Mismatch Summary',
        width: 750,
        height: 400
    };
    var chart = new google.visualization.PieChart(document.getElementById('topK_result_chart_div'));
    chart.draw(chartData, options);
}


function drawResultChart() {
    var chartData = new google.visualization.DataTable();
    chartData.addColumn('string', 'Top K');
    chartData.addColumn('number', 'Matchs');
    chartData.addRows([
        ['Matched 1st Choice', data.topCounts[0]],
        ['Matched 2nd Choice', data.topCounts[1]],
        ['Matched 3rd Choice', data.topCounts[2]],
        ['Matched 4th Choice', data.topCounts[3]],
        ['Matched 5th Choice', data.topCounts[4]]
    ]);
    var options = {
        title: 'Image Matches',
        width: 750,
        height: 400
    };
    var chart = new google.visualization.PieChart(document.getElementById('result_chart_div'));
    chart.draw(chartData, options);
}



document.addEventListener('DOMContentLoaded', function () {
    addResultHistory();

});


function addResultHistory() {

    combinedArrayData = [];
    combinedArrayData.push(['Model', 'Match', 'Mismatch']);


    modelHistories.forEach((model, index) => {

        combinedArrayData.push([model.modelName, model.passCount, model.totalMismatch])

        var divChild = document.createElement("div");
        divChild.id = 'model-history-' + index;
        divChild.className = "column-3";
        var div = document.getElementById('compare-result-graphs').appendChild(divChild)
        // $('#compare-result-graphs').append($('div').attr('id', 'model-history-' + index));

        google.charts.load('current', {
            'packages': ['bar']
        });

        google.charts.setOnLoadCallback(function () {

            $(function () {
                var chartData = google.visualization.arrayToDataTable([
                    ['  ', 'Match', 'Mismatch'],
                    ['Summary', model.passCount, model.totalMismatch]
                ]);
                var options = {
                    title: 'Generic Model Overall Result Summary',
                    vAxis: {
                        title: 'Images'
                    },
                    width: 400,
                    height: 400
                };
                var chart = new google.charts.Bar(document.getElementById('model-history-' + index));
                chart.draw(chartData, google.charts.Bar.convertOptions(options));
            });

        });

        google.charts.setOnLoadCallback(function () {
            var chartData = google.visualization.arrayToDataTable(combinedArrayData);
            var options = {
                title: 'Overall Result Summary',
                vAxis: {
                    title: 'Images'
                },
                width: 800,
                height: 600,
                bar: {
                    groupWidth: '30%'
                },
                isStacked: true,
                series: {
                    0: {
                        color: 'green'
                    },
                    1: {
                        color: 'Indianred'
                    }
                }
            };
            var chart = new google.charts.Bar(document.getElementById('Model_Stats_master'));
            chart.draw(chartData, google.charts.Bar.convertOptions(options));


        });







    })
}