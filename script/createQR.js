$(document).ready(
function () {
	var qrcode = new QRCode(document.getElementById("qrcode"), {
	width: 150,
	height: 150,
	colorDark : "#000000",
	colorLight : "#ffffff",
	correctLevel : QRCode.CorrectLevel.H
});

	qrcode.makeCode("https://www.vaultoro.com/images/advantages/face-book-bitcoin-gold-vaultoro.jpg");
});