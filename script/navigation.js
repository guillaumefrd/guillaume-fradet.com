
$(document).ready(function() {
	//change en EUR
	$.ajax({
		url : "https://blockchain.info/ticker?cors=true",
		dataType : "json",
		contentType : "application/json; charset=utf-8",
		type : "GET",
		timeout:	"5000",
		async : true,

		success : function(data) {
			var res = $('data')
			$('#bitcoin_change').append(data.EUR["15m"]); 
		},

		error : function(xhr, status, err) {
			$('#bitcoin_change').append(err+" N/A");
		}
	});

	//change en USD
	$.ajax({
		url : "https://blockchain.info/ticker?cors=true",
		dataType : "json",
		contentType : "application/json; charset=utf-8",
		type : "GET",
		timeout:	"5000",
		async : true,

		success : function(data) {
			$('#bitcoin_change_usd').append(data.USD["15m"]);
		},

		error : function(xhr, status, err) {
			$('#bitcoin_change_usd').append(err+" N/A");
		}
	});

	//nombre de block, minutes entre les blocks, revenus des mineurs, taille des blocks
	$.ajax({
		url : "https://api.blockchain.info/stats?cors=true",
		dataType : "json",
		type : "GET",
		timeout:	"5000",
		async : true,

		success : function(data) {
			$('#bitcoin_block_number').append(data.n_blocks_total);
			$('#bitcoin_minutes').append(data.minutes_between_blocks);
			$('#miners_revenue_btc').append(data.miners_revenue_btc);
			$('#blocks_size').append(data.blocks_size);
		},

		error : function(xhr, status, err) {
			$('#bitcoin_minutes').append(err+" N/A");
		}
	});

});

