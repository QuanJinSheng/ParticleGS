'use strict';
const $ = s => document.querySelector(s);
const $$ = s => [...document.querySelectorAll(s)];
const dialog=$('#figureDialog');
$$('[data-figure]').forEach(b=>b.addEventListener('click',()=>{
  $('#dialogImage').src=`assets/${b.dataset.figure}.webp`;
  $('#dialogImage').alt=b.dataset.caption;
  $('#figureCaption').textContent=b.dataset.caption;
  $('#dialogImageWrap').classList.remove('zoomed');
  $('#zoomFigure').textContent='Original size';
  $('#zoomFigure').setAttribute('aria-pressed','false');
  dialog.showModal();document.body.classList.add('dialog-open');
}));
$('#closeFigure').addEventListener('click',()=>dialog.close());
dialog.addEventListener('click',e=>{if(e.target===dialog)dialog.close();});
dialog.addEventListener('close',()=>document.body.classList.remove('dialog-open'));
$('#zoomFigure').addEventListener('click',()=>{
  const zoomed=$('#dialogImageWrap').classList.toggle('zoomed');
  $('#zoomFigure').textContent=zoomed?'Fit to window':'Original size';
  $('#zoomFigure').setAttribute('aria-pressed',zoomed);
});
$('#copyCitation').addEventListener('click',async()=>{
  const text=$('#citation').textContent;
  try {
    if(navigator.clipboard&&window.isSecureContext)await navigator.clipboard.writeText(text);
    else {
      const box=document.createElement('textarea');box.value=text;box.style.position='fixed';box.style.opacity='0';
      document.body.append(box);box.select();const ok=document.execCommand('copy');box.remove();if(!ok)throw new Error('copy');
    }
    $('#copyCitation').textContent='Copied';$('#copyStatus').textContent='BibTeX copied to clipboard.';
  }catch{
    $('#copyStatus').textContent='Select the citation below and copy it manually.';
    const r=document.createRange();r.selectNodeContents($('#citation'));const s=window.getSelection();s.removeAllRanges();s.addRange(r);
  }
});
const video=$('#demoVideo'),reduced=matchMedia('(prefers-reduced-motion: reduce)').matches;
if(reduced){video.autoplay=false;video.pause();}
new IntersectionObserver(entries=>{
  if(!entries[0].isIntersecting)video.pause();
  else if(!reduced)video.play().catch(()=>{});
},{threshold:.15}).observe(video);
document.addEventListener('visibilitychange',()=>{if(document.hidden)video.pause();});
