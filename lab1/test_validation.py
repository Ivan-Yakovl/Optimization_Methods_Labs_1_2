"""Регрессионные проверки. Запуск: python test_validation.py (также python -O)."""
import copy, json, os, tempfile, unittest
from pathlib import Path
import numpy as np
import common
import main as model

class ValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.lab=json.loads((common.ROOT/'metadata.json').read_text(encoding='utf-8'))['lab']
        cls.data=model.load_data()
        filename='base_milp.json' if cls.lab==3 else 'base.json'
        cls.result=json.loads((common.ROOT/'results'/filename).read_text(encoding='utf-8'))
        if cls.lab==1: cls.patterns=model.enumerate_patterns(cls.data)

    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix='lp_validation_')
        self.saved_common=common.OUT; self.saved_model=model.OUT; self.saved_cwd=Path.cwd()
        common.OUT=Path(self.tmp.name)/'results'; common.OUT.mkdir()
        model.OUT=common.OUT; os.chdir(self.tmp.name)

    def tearDown(self):
        os.chdir(self.saved_cwd); common.OUT=self.saved_common; model.OUT=self.saved_model
        self.tmp.cleanup()

    def verify(self,x,objective=None):
        objective=self.result['objective'] if objective is None else objective
        if self.lab==1: return model.verify(self.data,self.patterns,x,'probe',objective)
        return model.verify(self.data,x,'probe',objective)

    def test_original_solution(self):
        self.assertTrue(self.verify(self.result['x'])['feasible'])

    def test_solver_model_and_options_survive_optimized_python(self):
        result=common.solve([1.0],[[1.0]],[2.0],[2.0],[10.0],'smoke')
        self.assertEqual(result['status'],'Optimal')
        self.assertAlmostEqual(result['objective'],2.0)
        self.assertEqual(result['rows'],1)

    def test_nonfinite_solution_rejected(self):
        for value in [float('nan'),float('inf'),-float('inf')]:
            with self.subTest(value=value):
                x=list(self.result['x']); x[self.data['n'] if self.lab==3 else 0]=value
                with self.assertRaises(ValueError): self.verify(x)

    def test_wrong_vector_size_rejected(self):
        for x in [self.result['x'][:-1],self.result['x']+[0.0]]:
            with self.assertRaises(ValueError): self.verify(x)

    def test_objective_error_rejected(self):
        with self.assertRaises(ValueError): self.verify(self.result['x'],self.result['objective']+1)

    def test_negative_variable_rejected(self):
        x=list(self.result['x']); x[0]=-1
        with self.assertRaises(ValueError): self.verify(x)

    def test_domain_constraint_rejected(self):
        x=list(self.result['x'])
        if self.lab==1:
            x[next(i for i,z in enumerate(x) if z>1e-7)]+=.1
        elif self.lab==2:
            x[0]=self.data['capacity'][0][0]+1
        else:
            i=next(i for i,z in enumerate(x[:self.data['n']]) if z>.5 and x[self.data['n']+i]>0)
            x[i]=0
        with self.assertRaises(ValueError): self.verify(x)

    def test_binary_fraction_or_pattern_geometry(self):
        if self.lab==3:
            x=list(self.result['x']); i=next(i for i,z in enumerate(x[:self.data['n']]) if z>.5); x[i]=.5
            with self.assertRaises(ValueError): self.verify(x)
        elif self.lab==1:
            patterns=copy.deepcopy(self.patterns); patterns[0]['rows'][0]['counts'][0]=1000
            with self.assertRaises(ValueError): model.verify(self.data,patterns,self.result['x'],'geometry',self.result['objective'])
        else:
            x=list(self.result['x']); x[24]+=1
            with self.assertRaises(ValueError): self.verify(x)

if __name__=='__main__': unittest.main(verbosity=2)
