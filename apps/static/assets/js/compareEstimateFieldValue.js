$(function () {
  const TABLE_ELEMENT = $("#resultTable");
  TABLE_ELEMENT.LoadingOverlay("show");
  var table = TABLE_ELEMENT.DataTable({
    paging: true,
    lengthChange: false,
    searching: true,
    ordering: false,
    info: true,
    autoWidth: false,
    responsive: false,
    columnDefs: [
      { width: "7%", targets: [0] },
      { width: "7%", targets: [1] },
      { width: "15%", targets: [2] },
      { width: "8%", targets: [3] },
      { width: "15%", targets: [4] },
      { width: "", targets: [5] },
      { width: "7%", targets: [6] },
    ],
    scrollY: "500px",
    scrollCollapse: true,
    dom: "lrtip",
    initComplete: function () {
      this.api()
        .columns()
        .every(function () {
          let column = this;
          let title = column.footer().textContent;

          if (["Coder", "Translator", "Checked"].includes(title)) {
            var select = $(
              '<select class="form-control"><option value="">All</option></select>'
            )
              .appendTo($(column.footer()).empty())
              .on("change", function () {
                var val = DataTable.util.escapeRegex($(this).val());

                column.search(val ? "^" + val + "$" : "", true, false).draw();
              });

            // Add list of options
            column
              .data()
              .unique()
              .sort()
              .each(function (d, j) {
                if (d == "") return;
                select.append('<option value="' + d + '">' + d + "</option>");
              });
          } else {
            // Create input element
            $(
              '<input type="text" class="form-control" placeholder="' +
                title +
                '" />'
            )
              .appendTo($(column.footer()).empty())
              .on("keyup change clear", function () {
                if (column.search() !== this.value) {
                  column.search(this.value).draw();
                }
              });
          }
        });
        TABLE_ELEMENT.LoadingOverlay("hide");
    },
  });
});
