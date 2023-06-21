$(document).ready(function () {
  $.fn.index.init();
  $.fn.index.submitEstimate();
});

jQuery.fn.extend({
  index: {
    init: function () {
      $.fn.index.displayOption();
      $(document).on("change", 'select[name="tracker"]', function (e) {
        $.fn.index.displayOption();
      });

      // cal doc mod quantity
      $(document).on('focusout', 'input[name="doc_mod_quantity"]', function(e){
        let $this = $(this);
        let arr = $this.val().split(' ');
        let totalDocModQty = 0;
        if(arr.filter(Boolean).length < 2){
          return;
        }
        if(arr.length %2 != 0 && $this.val() != '') {
          return Swal.fire('doc_mod_quantity is not correct');
        }

        // process
        for (let index = 0; index < arr.length; index+=2) {
          totalDocModQty += Math.abs(parseInt(arr[index + 1]) - parseInt(arr[index])) + 1;
        }
        $this.val(totalDocModQty);
      })
    },
    displayOption: function(){
      const value = $('select[name="tracker"]').find("option:selected").val();
        if(value?.toLowerCase() == 'coding'){
          $('.js-coding-method-level').show();
        }
        else{
          $('.js-coding-method-level').hide();
        }
    },
    submitEstimate: function() {
      $(document).on('click', '#submitEstimate', function(e){
        e.preventDefault();
        const selectedValue = $('select[name=doc_file_format]').val()?.join(',');
        let data = $('#estimateForm').serializeJSON();
        data.doc_file_format = selectedValue;
        let required = Object.keys(data).find(x => data[x] == '' && x != 'doc_file_format');
        if(required){
          Swal.fire({
            icon: 'error',
            title: 'Error',
            text: `${required} is required`,
            scrollbarPadding: false,
            heightAuto: false,
            backdrop:false,
            didClose: () => $(`input[name='${required}']`)[0].focus()
          });
         
          return;
        }
        if(data.tracker.toLowerCase() != 'coding'){
          data.coding_method_level = 100;
        }
        $.ajax({
          url: '/calc',
          type: 'post',
          data: data,
          success: function( v ){
              // alert(`${v.result}  (Hours)`);
              return Swal.fire(`${v.result}  (Hours)`);

          },
          error: function( jqXhr, textStatus, errorThrown ){
              alert('Error');
          }
      });
      })
    }
  },
});
