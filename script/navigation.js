

$(document).ready(function() {
	$.ajax({
		url : "https://api.blockcypher.com/v1/btc/main",
		dataType : "json",
		contentType : "application/json; charset=utf-8",
		type : "GET",
		timeout:	"5000",
		async : false,

		success : function(data) {
			$('#bitcoin_block_number').append(data.height);
		},

		error : function(xhr, status, err) {
			$('#bitcoin_block_number').append(err+" N/A");
		}
	});
});

$(document).ready(function() {
	$.ajax({
		url : "https://blockchain.info/tobtc?currency=EUR&value=1",
		dataType : "html",
		contentType : "application/json; charset=utf-8",
		type : "GET",
		timeout:	"5000",
		async : false,

		success : function(data) {
			$('#bitcoin_change').append((1/data).toFixed(3)); //inverse pour avoir dans l'autre sens de change
		},

		error : function(xhr, status, err) {
			$('#bitcoin_change').append(err+" N/A");
		}
	});
});

$(document).ready(function() {
	$.ajax({
		url : "https://blockchain.info/tobtc?currency=USD&value=1",
		dataType : "json",
		contentType : "application/json; charset=utf-8",
		type : "GET",
		timeout:	"5000",
		async : false,

		success : function(data) {
			$('#bitcoin_change_usd').append((1/data).toFixed(3));
		},

		error : function(xhr, status, err) {
			$('#bitcoin_change_usd').append(err+" N/A");
		}
	});
});

$(document).ready(function() {
	$.ajax({
		url : "https://api.blockchain.info/stats",
		dataType : "json",
		contentType : "application/json; charset=utf-8",
		type : "GET",
		timeout:	"5000",
		async : false,

		success : function(data) {
			$('#bitcoin_minutes').append(data);
		},

		error : function(xhr, status, err) {
			$('#bitcoin_minutes').append(err+" N/A");
		}
	});
});
/*
var settings = {
  "async": true,
  "crossDomain": true,
  "url": "https://api.blockchain.info/stats",
  "method": "GET",
  "headers": {
    "cache-control": "no-cache",
    "postman-token": "16b6a853-ce1b-0fb9-8364-5a79c194e49a"
  }
}

$.ajax(settings).done(function (response) {
  alert(response);
});

*/