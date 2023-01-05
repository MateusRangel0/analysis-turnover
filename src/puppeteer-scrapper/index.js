const puppeteer = require("puppeteer");
const chalk = require("chalk");
var fs = require("fs");

// MY OCD of colorful console.logs for debugging... IT HELPS
const error = chalk.bold.red;
const success = chalk.keyword("green");
const lastPageNumber = 35;

empresas = ['accenture', 'stefanini', 'ibm', 'totvs', 'algar-tech', 'sonda-it', 'hp-inc', 'dell', 'linx', 'oracle', 'resource-it-solutions', 'ericsson', 'vivo-telefonica-brasil', 'tivit', 'tim', 'claro-brasil', 'b2w-digital', 'concentrix', 'nextel-telecomunicacoes'];

(async () => {
  try {
    // open the headless browser
    var browser = await puppeteer.launch({headless: false });
    // open a new page
    var page = await browser.newPage();
    await page.goto(`https://www.lovemondays.com.br/`, {waitUntil: 'networkidle2'});
    await page.waitFor(20000);
    for (let j = 0; j <= empresas.length; j++) {
      console.log(empresas[j]);      
      fs.mkdir('data/' + empresas[j], { recursive: true }, (err) => {
        if (err) throw err;
      });      
      for (let index = 1; index <= lastPageNumber; index++) {      
        await page.goto(`https://www.lovemondays.com.br/trabalhar-na-` + empresas[j] + `/avaliacoes/pagina/` +index);
        await page.waitForSelector("ul.lm-List-default");
        var ratings = await page.evaluate(() => {
          var job = document.querySelectorAll(`span.reviewer`);
          var avaliation = document.querySelectorAll(`div.lm-Review-contribution`);
          var titleLinkArray = [];
          for (var i = 0; i < job.length; i++) {
              titleLinkArray[i] = {
                user: job[i].innerText,
                avaliation: avaliation[i].innerText
              };
          }
          return titleLinkArray;
        });
        fs.writeFile("data/" + empresas[j] + "/" + index + ".json", JSON.stringify(ratings), function(err) {
          if (err) throw err;
          console.log(index);
        });
      }
    }
    await browser.close();
  } catch (err) {
    // Catch and display errors
    console.log(error(err));
    await browser.close();
    console.log(error("Browser Closed"));
  }
})();