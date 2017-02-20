
var xmlhttp = new XMLHttpRequest();
xmlhttp.onreadystatechange = function() {
    if (this.readyState == 4 && this.status == 200) {
        var myObj = this.responseText;
        var jsonPretty = JSON.stringify(JSON.parse(myObj),null,2);
        document.getElementById("info").innerHTML = jsonPretty;
    }
};
xmlhttp.open("GET", "http://bitcoin.mubiz.com/info", true);
xmlhttp.send();

var xmlhttp = new XMLHttpRequest();
xmlhttp.onreadystatechange = function() {
    if (this.readyState == 4 && this.status == 200) {
        var myObj = this.responseText;
        var jsonPretty = JSON.stringify(JSON.parse(myObj),null,2);
        document.getElementById("blockchaininfo").innerHTML = jsonPretty;
    }
};
xmlhttp.open("GET", "http://bitcoin.mubiz.com/blockchaininfo", true);
xmlhttp.send();

var xmlhttp = new XMLHttpRequest();
xmlhttp.onreadystatechange = function() {
    if (this.readyState == 4 && this.status == 200) {
        var myObj = this.responseText;
        var jsonPretty = JSON.stringify(JSON.parse(myObj),null,2);
        document.getElementById("mininginfo").innerHTML = jsonPretty;
    }
};
xmlhttp.open("GET", "http://bitcoin.mubiz.com/mininginfo", true);
xmlhttp.send();

var xmlhttp = new XMLHttpRequest();
xmlhttp.onreadystatechange = function() {
    if (this.readyState == 4 && this.status == 200) {
        var myObj = this.responseText;
        var jsonPretty = JSON.stringify(JSON.parse(myObj),null,2);
        document.getElementById("peerinfo").innerHTML = jsonPretty;
    }
};
xmlhttp.open("GET", "http://bitcoin.mubiz.com/peerinfo", true);
xmlhttp.send();