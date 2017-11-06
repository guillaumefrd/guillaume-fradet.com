function Query()
{
    var adress = document.getElementById("adressInput").value;
    adress = adress.replace(/ /g,""); //supprime les espaces pour ne pas les prendre en compte
    var select = document.getElementById("sel1").value;

    if(select == "adress")
    {
        var isAddress = /^[0-9a-zA-Z]{34}$/.test(adress);
        if(isAddress == false)
        {
            document.getElementById("result").innerHTML = " <p class=\"text-danger\"><span class=\"glyphicon glyphicon-remove\"></span> Invalid input <span class=\"glyphicon glyphicon-remove\"></span></p>"
            document.getElementById("resultDiv").style.visibility = "visible";
        }

        else
            AdressQuery();
    }

    if(select == "hashTransaction")
    {
        var isHash  = /^[0-9A-F]{64}$/i.test(adress);
        if(isHash == false || adress[0] == "0")
        {
            document.getElementById("result").innerHTML = " <p class=\"text-danger\"><span class=\"glyphicon glyphicon-remove\"></span> Invalid input <span class=\"glyphicon glyphicon-remove\"></span></p>"
            document.getElementById("resultDiv").style.visibility = "visible";
        }

        else
            HashTransactionQuery();
    }

    if(select == "hashBlock")
    {
        var isHash  = /^[0-9A-F]{64}$/i.test(adress);
        if(isHash == false || adress[0] != "0")
        {
            document.getElementById("result").innerHTML = " <p class=\"text-danger\"><span class=\"glyphicon glyphicon-remove\"></span> Invalid input <span class=\"glyphicon glyphicon-remove\"></span></p>"
            document.getElementById("resultDiv").style.visibility = "visible";
        }

        else
            HashBlocQuery();
    }

    if(select == "indexBlock")
    {
        var length = adress.length;
        var isNumber = /(^[0-9]{1}$)|(^[0-9]{2}$)|(^[0-9]{3}$)|(^[0-9]{4}$)|(^[0-9]{5}$)|(^[0-9]{6}$)/.test(adress);
        if(isNumber == false)
        {
            document.getElementById("result").innerHTML = " <p class=\"text-danger\"><span class=\"glyphicon glyphicon-remove\"></span> Invalid input <span class=\"glyphicon glyphicon-remove\"></span></p>"
            document.getElementById("resultDiv").style.visibility = "visible";
        }
        else
            IndexBlocQuery();
    }
}

function AdressQuery()
{
    var xmlhttp = new XMLHttpRequest();
    xmlhttp.onreadystatechange = function() {
        if (this.readyState == 4 && this.status == 200) {
            var myObj = this.responseText;
            var jsonPretty = JSON.stringify(JSON.parse(myObj),null,2);
            jsonPretty = syntaxHighlight(jsonPretty);
            document.getElementById("result").innerHTML = jsonPretty;
        }
    };
    var adress = document.getElementById("adressInput").value;
    var strAdress = "https://bitcoin.mubiz.com/address/bitcoin_address/" + adress + "/";
    xmlhttp.open("GET", strAdress, true);
    xmlhttp.send();
    document.getElementById("resultDiv").style.visibility = "visible";
}

function HashTransactionQuery()
{
    var xmlhttp = new XMLHttpRequest();
    xmlhttp.onreadystatechange = function() {
        if (this.readyState == 4 && this.status == 200) {
            var myObj = this.responseText;
            var jsonPretty = JSON.stringify(JSON.parse(myObj),null,2);
            jsonPretty = syntaxHighlight(jsonPretty);
            document.getElementById("result").innerHTML = jsonPretty;

        }
    };
    var adress = document.getElementById("adressInput").value;
    var strAdress = "https://bitcoin.mubiz.com/transaction/" + adress + "/";
    xmlhttp.open("GET", strAdress, true);
    xmlhttp.send();
    document.getElementById("resultDiv").style.visibility = "visible";
}

function HashBlocQuery()
{
    var xmlhttp = new XMLHttpRequest();
    xmlhttp.onreadystatechange = function() {
        if (this.readyState == 4 && this.status == 200) {
            var myObj = this.responseText;
            var jsonPretty = JSON.stringify(JSON.parse(myObj),null,2);
            jsonPretty = syntaxHighlight(jsonPretty);
            document.getElementById("result").innerHTML = jsonPretty;

        }
    };
    var adress = document.getElementById("adressInput").value;
    var strAdress = "https://bitcoin.mubiz.com/block_hash/" + adress + "/";
    xmlhttp.open("GET", strAdress, true);
    xmlhttp.send();
    document.getElementById("resultDiv").style.visibility = "visible";
}

function IndexBlocQuery()
{
    var xmlhttp = new XMLHttpRequest();
    xmlhttp.onreadystatechange = function() {
        if (this.readyState == 4 && this.status == 200) {
            var myObj = this.responseText;
            var jsonPretty = JSON.stringify(JSON.parse(myObj),null,2);
            jsonPretty = syntaxHighlight(jsonPretty);
            document.getElementById("result").innerHTML = jsonPretty;

        }
    };
    var adress = document.getElementById("adressInput").value;
    var strAdress = "https://bitcoin.mubiz.com/block_index/" + adress + "/";
    xmlhttp.open("GET", strAdress, true);
    xmlhttp.send();
    document.getElementById("resultDiv").style.visibility = "visible";
}

function syntaxHighlight(json) {
    json = json.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
    return json.replace(/("(\\u[a-zA-Z0-9]{4}|\\[^u]|[^\\"])*"(\s*:)?|\b(true|false|null)\b|-?\d+(?:\.\d*)?(?:[eE][+\-]?\d+)?)/g, function (match) {
        var cls = 'number';
        if (/^"/.test(match)) {
            if (/:$/.test(match)) {
                cls = 'key';
            } else {
                cls = 'string';
            }
        } else if (/true|false/.test(match)) {
            cls = 'boolean';
        } else if (/null/.test(match)) {
            cls = 'null';
        }
        return '<span class="' + cls + '">' + match + '</span>';
    });
}

function tryItAdress()
{
    var example = $('#ExampleAdress').attr('value');
    $('#adressInput').attr('value',example);
    $('#sel1').val('adress');
}

function tryItHashTr()
{
    var example = $('#ExampleHashTr').attr('value');
    $('#adressInput').attr('value',example);
    $('#sel1').val('hashTransaction');
}

function tryItHashBlk()
{
    var example = $('#ExampleHashBlk').attr('value');
    $('#adressInput').attr('value',example);
    $('#sel1').val('hashBlock');
}

function tryItIndex()
{
    var example = $('#ExampleIndex').attr('value');
    $('#adressInput').attr('value',example);
    $('#sel1').val('indexBlock');
}
