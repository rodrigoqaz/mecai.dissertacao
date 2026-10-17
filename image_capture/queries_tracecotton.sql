WITH classificacoes AS (
    SELECT
        f.codigosai 
        ,ac.descricao AS classificacao
        ,'lote' AS origem
        ,CAST(ml.numero AS varchar) AS numero
        ,am.nome AS maquina_algodoeira
        ,faz.nome AS fazenda_origem
        ,f.databeneficiamento
        ,r.datacolheita
        ,r.periodocolheita
        ,r.bordadura
        ,r.umidade
        ,r.sinistros
        ,r.contaminantes
        ,r.localizacaolatitude
        ,r.localizacaolongitude
        ,r.localizacaoaltitude
        ,t.numero AS talhao
        ,t.idexterno AS idexterno
        ,t.areaha
        ,t.dataplantio
        ,v.nome AS nome_variedade
        ,v.descricao AS descricao_variedade
        ,g.nome AS nome_grupovariedade
        ,g.descricao AS descricao_grupovariedade
    FROM tracecotton.fardinho f
    INNER JOIN tracecotton.algodoeiramaquina am ON (am.algodoeiramaquinaid = f.algodoeiramaquinaid)
    INNER JOIN tracecotton.fazenda faz ON (faz.fazendaid = f.fazendaorigemid)
    INNER JOIN tracecotton.rolo r ON (r.roloid = f.roloid)
    INNER JOIN tracecotton.ordemcolheita oc ON (oc.ordemcolheitaid = r.ordemcolheitaid)
    INNER JOIN tracecotton.talhao t ON (t.talhaoid = oc.talhaoid)
    INNER JOIN tracecotton.variedade v ON (v.variedadeid = t.variedadeid)
    INNER JOIN tracecotton.grupovariedade g ON (g.grupovariedadeid = v.grupovariedadeid)
    INNER JOIN tracecotton.malaamostralote ml ON (ml.malaamostraloteid = f.malaamostraloteid)
    INNER JOIN tracecotton.algodaoclassificacao ac ON (ac.algodaoclassificacaoid = ml.algodaoclassificacaoid)
    
    UNION all

    SELECT
        f.codigosai 
        ,ac.descricao AS classificacao
        ,'mala' AS origem
        ,mai.numero AS numero
        ,am.nome AS maquina_algodoeira
        ,faz.nome AS fazenda_origem
        ,f.databeneficiamento
        ,r.datacolheita
        ,r.periodocolheita
        ,r.bordadura
        ,r.umidade
        ,r.sinistros
        ,r.contaminantes
        ,r.localizacaolatitude
        ,r.localizacaolongitude
        ,r.localizacaoaltitude
        ,t.numero AS talhao
        ,t.idexterno AS idexterno
        ,t.areaha
        ,t.dataplantio
        ,v.nome AS nome_variedade
        ,v.descricao AS descricao_variedade
        ,g.nome AS nome_grupovariedade
        ,g.descricao AS descricao_grupovariedade
    FROM tracecotton.fardinho f
    INNER JOIN tracecotton.algodoeiramaquina am ON (am.algodoeiramaquinaid = f.algodoeiramaquinaid)
    INNER JOIN tracecotton.fazenda faz ON (faz.fazendaid = f.fazendaorigemid)
    INNER JOIN tracecotton.rolo r ON (r.roloid = f.roloid)
    INNER JOIN tracecotton.ordemcolheita oc ON (oc.ordemcolheitaid = r.ordemcolheitaid)
    INNER JOIN tracecotton.talhao t ON (t.talhaoid = oc.talhaoid)
    INNER JOIN tracecotton.variedade v ON (v.variedadeid = t.variedadeid)
    INNER JOIN tracecotton.grupovariedade g ON (g.grupovariedadeid = v.grupovariedadeid)
    INNER JOIN tracecotton.fardinhoamostraintermediaria fai ON (fai.codigosai = f.codigosai AND fai.organizacaoid = f.organizacaoid)
    INNER JOIN tracecotton.malaamostraintermediaria mai ON (fai.malaamostraid = mai.malaamostraintermediariaid)
    INNER JOIN tracecotton.algodaoclassificacao ac ON (ac.algodaoclassificacaoid = mai.algodaoclassificacaoid)  
), temp AS (
SELECT 
    *
    ,ROW_NUMBER() OVER(PARTITION BY codigosai, classificacao ORDER BY codigosai, origem) AS rn
FROM classificacoes
)
SELECT * FROM temp WHERE rn=1



-- Verifica onde que estão as malas
SELECT
	la.descricao
	,mai.numero 
	,ac.descricao 
	,count(DISTINCT(fai.codigosai))
FROM tracecotton.malaamostraintermediaria mai
INNER JOIN tracecotton.localarmazenamento la ON (la.localarmazenamentoid = mai.localarmazenamentoid)
INNER JOIN tracecotton.algodaoclassificacao ac ON (ac.algodaoclassificacaoid = mai.algodaoclassificacaoid)
INNER JOIN tracecotton.fardinhoamostraintermediaria fai ON (fai.malaamostraid = mai.malaamostraintermediariaid)
WHERE 1=1
	AND mai.algodoeiraid = '2d264d07-010d-8d51-813f-01904f60d682'
	--AND la.descricao = 'SALA DE CLASSIFICAÇÃO'
GROUP BY la.descricao,mai.numero,ac.descricao 

	

