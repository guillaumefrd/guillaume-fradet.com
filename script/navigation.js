

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
			$('#bitcoin_change').append(data); //inverse pour avoir dans l'autre sens de change
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
			$('#bitcoin_change_usd').append(data);
		},

		error : function(xhr, status, err) {
			$('#bitcoin_change_usd').append(err+" N/A");
		}
	});
});
