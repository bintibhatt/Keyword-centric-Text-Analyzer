$(document).ready(function() {
    $('#search-button').click(function() {
        var text = $('#text-input').val();
        var keywords = $('#keywords-input').val().split(',');

        var searchData = {
            text: text,
            keywords: keywords
        };

        $.ajax({
            url: '/search',
            type: 'POST',
            contentType: 'application/json',
            data: JSON.stringify(searchData),
            success: function(response) {
                var sentences = response.sentences;
                var resultsList = $('#results-list');
                resultsList.empty();

                if (sentences.length > 0) {
                    for (var i = 0; i < sentences.length; i++) {
                        var listItem = $('<li>').text(sentences[i]);
                        resultsList.append(listItem);
                    }
                } else {
                    var listItem = $('<li>').text('No sentences found.');
                    resultsList.append(listItem);
                }
            },
            error: function(error) {
                console.log('Error:', error);
            }
        });
    });
});
