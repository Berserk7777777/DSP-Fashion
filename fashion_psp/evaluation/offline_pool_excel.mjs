// Local artifact-tool backend. Python uses openpyxl only on servers without it.
import fs from 'node:fs/promises';
import { createRequire } from 'node:module';
import { pathToFileURL } from 'node:url';
const require = createRequire(import.meta.url);
const { Workbook, SpreadsheetFile } = await import(pathToFileURL(require.resolve('@oai/artifact-tool')).href);
const spec = JSON.parse(await fs.readFile(process.argv[2], 'utf8'));
const book = Workbook.create();
for (const table of spec.tables) {
  const sheet = book.worksheets.add(table.name);
  sheet.showGridLines = false;
  const rows = table.rows.map(row => row.map(v => typeof v === 'string' && v.startsWith('=') ? "'" + v : v));
  const n = rows.length, m = rows[0].length;
  sheet.getRangeByIndexes(0, 0, n, m).values = rows;
  sheet.getRangeByIndexes(0, 0, n, m).format.font = { name: 'Arial', size: 10 };
  sheet.getRangeByIndexes(0, 0, n, m).format.columnWidth = 18;
  sheet.getRangeByIndexes(0, 0, 1, m).format = {
    fill: '#243746', font: { name: 'Arial', size: 10, bold: true, color: '#FFFFFF' }, rowHeight: 34, wrapText: true,
  };
  for (let col = 0; col < m; col++) {
    const name = rows[0][col];
    sheet.getRangeByIndexes(0, col, n, 1).format.columnWidth = Math.max(18, Math.min(34, name.length+2));
    if (/path|definition|population|sha256|signature|image_kind/.test(name)) {
      sheet.getRangeByIndexes(0, col, n, 1).format.columnWidth = name === 'definition' ? 100 : 36;
    }
    if (/^(cs|cis|lpips|personal|compatibility)(_|$)/.test(name) && !/(_rank|_n)$/.test(name) && n > 1) {
      sheet.getRangeByIndexes(1, col, n-1, 1).setNumberFormat('0.000000');
    }
  }
  sheet.freezePanes.freezeRows(1);
  if (table.name === 'PSP_trajectories') sheet.freezePanes.freezeColumns(8);
  if (n > 1) sheet.tables.add(sheet.getRangeByIndexes(0, 0, n, m), true, `T_${table.name}`);
}
book.recalculate();
const temp = spec.output + '.tmp.xlsx';
await (await SpreadsheetFile.exportXlsx(book)).save(temp);
await fs.rename(temp, spec.output);
if (spec.preview) {
  console.log((await book.inspect({kind:'region',sheetId:'Summary',range:'A1:G6',maxChars:1500,tableMaxRows:6,tableMaxCols:7})).ndjson);
  const preview = await book.render({sheetName:'Coverage',autoCrop:'all',scale:1,format:'png'});
  await fs.writeFile(spec.output.replace(/\.xlsx$/, '_preview.png'), new Uint8Array(await preview.arrayBuffer()));
}
