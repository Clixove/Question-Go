function check_all(button, class_name) {
    const cbs = document.getElementsByName(class_name);
    if (button.checked) {
        cbs.forEach((cb) => cb.checked = true);
    } else {
        cbs.forEach((cb) => cb.checked = false);
    }
}
function copy_selected_instances(selected_table, target_id) {
    const target_button = document.getElementById(target_id);
    while (target_button.firstChild) {
        target_button.removeChild(target_button.firstChild);
    }
    const checkboxes = document.getElementsByName(selected_table);
    const checkboxes_checked = [];
    for (let j=0; j<checkboxes.length; j++) {
        if (checkboxes[j].checked) {
            let copy_checkbox = checkboxes[j].cloneNode(true);
            checkboxes_checked.push(copy_checkbox);
        }
    }
    for (let j=0; j<checkboxes_checked.length; j++) {
        target_button.appendChild(checkboxes_checked[j]);
    }
}
function async_submit_form(form_id, url, response_id) {
    $('#' + form_id).submit(function (e) {
        e.preventDefault();
        // document.getElementById(response_id).innerHTML = '' +
        //     '<div class="alert alert-warning alert-dismissible fade show" role="alert">' +
        //     '    <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>' +
        //     '    Start running ...' +
        //     '</div>';
        $.ajax({
            type: 'POST',
            url: url,
            data: $(this).serialize(),
            success: (response) => {document.getElementById(response_id).innerHTML = response},
        });
    });
}
function export_html(id, filename) {
    const text = document.getElementById(id).innerHTML;
    if (text.length > 0) {
        const blob = new Blob([text], {type: "text/plain;charset=utf-8"});
        const link = document.createElement('a');
        link.download = filename;
        link.href = window.URL.createObjectURL(blob);
        link.click();
        window.URL.revokeObjectURL(link.href);
    }
}
function set_compact(id) {
    const ul = document.getElementById(id);
    ul.setAttribute("class", "d-flex flex-wrap");
    const li = ul.children;
    for (let x=0; x<li.length; x++) {
        li[x].setAttribute("class", "flex-item text-nowrap");
        li[x].setAttribute("style", "width: 14rem; overflow-x: hidden;");
    }
}
function set_expand(id) {
    const ul = document.getElementById(id);
    ul.removeAttribute("class");
}
