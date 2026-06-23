import casadi as ca

from npmpc.dynamics.integrators import IntegrationBase

class MPCBase:
    params: dict
    integrator: IntegrationBase
    p: ca.DM
    slack_weight = 1000
    
    def cost_function(self, x: ca.MX, u: ca.MX) -> ca.MX:
        """MPC cost; subclasses read cost weights from self.params['cost']."""
        raise NotImplementedError


    def integration_constraints(self, x: ca.MX, u: ca.MX, dt: float) -> list:
        """Dynamics integration constraints over the horizon."""
        return self.integrator.casadi_implicit(x, u, dt, self.p)


    def slack_constraints(self, slack: ca.MX, slack_bound: dict) -> list:
        """Bounds on the slack variables."""
        constraints = []; x_bound = slack_bound['x']
        for i, bound in enumerate(x_bound):
            if bound[0] != ca.inf: 
                constraints.append(slack[i] >= 0)
                constraints.append(slack[i] <= bound[0])

        return constraints
    
    
    def slack_cost(self, slack: ca.MX, slack_bound: dict) -> ca.MX:
        """Penalty cost on the slack variables."""
        cost = 0; x_bound = slack_bound['x']
        for i, bound in enumerate(x_bound):
            if bound[0] != ca.inf:
                cost += self.slack_weight * 0.5 * (slack[i]**2 + slack[i])

        return cost
        

    def state_constraints(self, x: ca.MX, slack: ca.MX, params: dict) -> list[ca.MX]:
        """State bound constraints, softened by slack where allowed."""
        constraints = []
        x_bound = params['hard_bound']['x']; slack_bound = params['slack_bound']['x']
        
        for i in range(x.shape[1]):
            if x_bound[i][0] != -ca.inf:
                if slack_bound[i] != ca.inf:
                    constraints.append(x[:,i] >= x_bound[i][0] - slack[i])
                else:
                    constraints.append(x[:,i] >= x_bound[i][0])
            if x_bound[i][1] != ca.inf:
                if slack_bound[i] != ca.inf:
                    constraints.append(x[:,i] <= x_bound[i][1] + slack[i])
                else:
                    constraints.append(x[:,i] <= x_bound[i][1])
        return constraints
    
    
    def input_constraints(self, u: ca.MX, params: dict) -> list[ca.MX]:
        """Input bound constraints."""
        constraints = []; u_bound = params['hard_bound']['u']
        for i in range(u.shape[1]):
            if u_bound[i][0] != -ca.inf: 
                constraints.append(u[:,i] >= u_bound[i][0])
            if u_bound[i][1] != ca.inf: 
                constraints.append(u[:,i] <= u_bound[i][1])
        return constraints