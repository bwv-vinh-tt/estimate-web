function getData() {
  const array = [];
  const label = ['Create Screen/API/Batch Qty', 'New/Mod', 'Doc Mod Quantity', 'Authority Qty',
      'Display/Output Items Qty', 'Validation Items Qty', 'Event row', 'Event Items Qty', 'Get API Qty', 'Create API Qty',
      'Update API Qty', 'Delete API Qty', 'Get Table Qty', 'Create Table Qty', 'Update Table Qty', 'Delete Table Qty', 'Doc Layout',
      'Doc Layout', 'Doc Understandable', 'Doc File Format', 'Business Logic Level', 'Coding Method Level', 'Spent time'];

  label.forEach(l => {
      if (l != 'Spent time') {
          var x = $($(`span:contains(${l})`).get(0)).parent().next().text();
          if (!x) {
              x = 0;
          }
          if (x == 'New') {
              x = 100
          }
          if (x == 'Mod') {
              x = 50
          }

          if (x == 'BW') {
              x = 1
          }

          if (x == 'Non-BW(Readable)') x = 2

          if (x == 'Non-BW(Not Readable)') x = 3

          if (x == 'MarkDown') x = 1

          if (x == 'Google Sheet') x = 2

          if (x == 'Excel') x = 3

          if (x == 'Other') x = 4

          if (typeof x == 'string' && x.includes('%')) {
              x = 100 - x.substring(0, x.length - 1);
          }
      } else {
          var x = $('.spent-time').find('div > a').text();
          if (typeof x == 'string' && x.includes('h')) {
              x = x.replace(' h', "");
          }
      }

      array.push(x);
  });
  return array.join(',');
}

getData()