'!TITLE "<Title>"
PROGRAM kretanje3d_inc2
takearm
motor on

'pomeraj
dim pomeraj_duz as integer
pomeraj_duz=10
dim pomeraj_ugao as integer
pomeraj_ugao=10

'dim korak as integer
dim xmax as integer
dim xmin as integer
dim ymax as integer
dim ymin as integer
dim zmax as integer
dim zmin as integer
dim zrmax as integer
dim zrmin as integer
dim vakuum as integer

vakuum=0




xmax=14
xmin=-170
ymax=750
ymin=500
zmin=0
zmax=400
zrmin=-135
zrmax=-45

dim x_zero as integer
dim y_zero as integer
dim z_zero as integer
dim xr_zero as integer
dim yr_zero as integer
dim zr_zero as integer
x_zero=int(PosX(p[6]))
y_zero=int(PosY(p[6]))
z_zero=int(PosZ(p[6]))
xr_zero=int(PosRX(p[6]))
yr_zero=int(PosRY(p[6]))
zr_zero=int(PosRZ(p[6]))
dim x_tren as integer
dim y_tren as integer
dim z_tren as integer
dim zr_tren as integer


move p,(xmax,ymin,z_zero,xr_zero,yr_zero,zr_zero)
x_tren=xmax
y_tren=ymin
z_tren=z_zero
zr_tren=zr_zero

dim ok as integer
ok=8
dim nok as integer
nok=0

dim greska as integer 
greska=false

COM_STATE #8, I[40]
IF I[40] = -1 THEN 
COM_ENCOM #8
DELAY 10
FLUSH #8

ENDIF



do while ( greska=false)

LINPUTB #8, I[10],1
FLUSH #8

'MAX
IF I[10] = 47 THEN 
x_tren=xmax
y_tren=ymin
ENDIF
'ZERO
IF I[10] = 48 THEN 
x_tren=x_zero
y_tren=y_zero
ENDIF
'X++
IF I[10] = 49 THEN 
if x_tren<=xmax THEN
x_tren=x_tren+pomeraj_duz
'printb #8, ok 'izlazis van tolerancija
'greska=true
ENDIF
ENDIF
'X--
IF I[10] =50 THEN 

if x_tren>=xmin THEN
x_tren=x_tren-pomeraj_duz
'printb #8, ok 'izlazis van tolerancija
'greska=true
ENDIF
ENDIF

'Y++
IF I[10] =51 THEN 
if y_tren<=ymax THEN
y_tren=y_tren+pomeraj_duz
'printb #8, ok 'izlazis van tolerancija
'greska=true
ENDIF
ENDIF
'Y--
IF I[10] = 52 THEN 
if y_tren>=ymin THEN
y_tren=y_tren-pomeraj_duz
'printb #8, ok 'izlazis van tolerancija
'greska=true
ENDIF
ENDIF
'Z++
IF I[10] =53 THEN 
if z_tren<=zmax  THEN
z_tren=z_tren+pomeraj_duz
'printb #8, ok 'izlazis van tolerancija
'greska=true
ENDIF
ENDIF
'Z--
IF I[10] = 54 THEN 
if z_tren>=zmin  THEN
z_tren=z_tren-pomeraj_duz
'printb #8, ok 'izlazis van tolerancija
'greska=true
ENDIF
ENDIF
'RZ++
IF I[10] =55 THEN 
if  zr_tren<=zrmax THEN
zr_tren=zr_tren+pomeraj_ugao
'printb #8, ok 'izlazis van tolerancija
'greska=true
ENDIF
ENDIF
'RZ--
IF I[10] =56 THEN 
if zr_tren>=zrmin  THEN
zr_tren=zr_tren-pomeraj_ugao
'printb #8, ok 'izlazis van tolerancija
'greska=true
ENDIF
ENDIF
'VACUUM ISKLJ.
IF I[10] = 45   THEN 
if vakuum=1  THEN
vakuum=0
set io26
reset io25
ENDIF
ENDIF
'VACUUM UKLJ.
IF I[10] = 43  THEN 
if vakuum=0  THEN
vakuum=1
reset io26
set io25
ENDIF
ENDIF
'i[15]=x_tren
'i[16]=y_tren
'i[17]=z_tren

IF I[10] <> 43 and I[10] <> 45 THEN 
move p,(x_tren,y_tren,z_tren,xr_zero,yr_zero,zr_tren)
ENDIF


FLUSH #8
printb #8, ok 'izvrsena komanda
'ENDIF


loop 

END
